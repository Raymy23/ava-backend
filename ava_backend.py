import os
import json
import datetime
import threading
import tempfile
import base64
import time

# Externí knihovny
import numpy as np 
import requests 
from flask import Flask, request, jsonify, send_from_directory # Zahrnuje send_from_directory
from flask_cors import CORS

# Externí knihovny pro API
from google import genai
from google.genai import types

# --- GLOBÁLNÍ KONFIGURACE ---
client = None 
chat = None  
MODEL_NAME = 'gemini-2.5-flash'
GEMINI_EMBEDDING_MODEL = 'text-embedding-004' 
MEMORY_FILE = 'ava_memory.json'
LISS_MEMORY = [] # Interní název pro paměť (může zůstat LISS_MEMORY)

LOG_FILE = 'ava_log.txt'
eleven_key = None 
VOICE_ID = "2Lb1en5ujrODDIqmp7F3" 
TTS_API_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"

# OPRAVA: Definování absolutní cesty k adresáři
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

AVA_PERSONA = (
    "Jsi AI asistentka jménem Ava. Vystupuješ jako slečna, která je příjemná, "
    "přátelská a vstřícná. Tvůj úkol je vést konverzaci a zároveň si pamatovat "
    "osobní fakta o uživateli (např. jeho preference, vybavení, apod.), které ti "
    "budou předány v kontextu 'DLOUHODOBÁ PAMĚŤ'. Odpovědi formuluj tak, abys "
    "informace z paměti využila přirozeně, ale neotravně. Vždy se drž své role."
)

# --- FUNKCE PRO ZÁPIS LOGU (Vždy PŘIDÁVÁ) ---
def zapis_log(zprava):
    global eleven_key 
    cas = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_zprava = f"{cas} {zprava}"
    try:
        # Tato funkce nyní VŽDY jen připojuje ('a')
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_zprava + '\n')
    except Exception:
        pass 
    print(log_zprava) # Vždy tiskneme do konzole


# --- FUNKCE PRO SMART SAVE (beze změny) ---
def analyze_for_fact(dotaz: str) -> dict:
    if not client:
        return {"should_save": False, "extracted_fact": "AI klient není aktivní."}

    SMART_SAVE_PROMPT = f"""
    Analyzuj následující zprávu od uživatele. Rozhodni, zda zpráva obsahuje osobní,
    trvalý fakt (např. jméno, preference, povolání, koníček, vybavení jako 'volant 1080 stupňů')
    nebo jen dotaz/komentář.

    Pokud je to fakt, nastav 'should_save' na true a extrahuj stručný fakt do 'extracted_fact'.
    Pokud to není fakt hodný uložení, nastav 'should_save' na false a 'extracted_fact' na prázdný řetězec.

    Zpráva: "{dotaz}"

    Vrať odpověď POUZE jako validní JSON objekt.
    Příklad 1: {{"should_save": true, "extracted_fact": "Uživatel se jmenuje Jakub a říká mu Kubo."}}
    Příklad 2: {{"should_save": false, "extracted_fact": ""}}
    """

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[SMART_SAVE_PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "should_save": {"type": "boolean"},
                        "extracted_fact": {"type": "string"},
                    }
                }
            )
        )
        return json.loads(response.text)

    except Exception as e:
        zapis_log(f"Chyba analýzy pro Smart Save: {e}")
        return {"should_save": False, "extracted_fact": f"Chyba analýzy: {e}"}


# --- FUNKCE PRO MANIPULACI S PAMĚTÍ A RAG (beze změny) ---

def nacti_pamet():
    global LISS_MEMORY
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                LISS_MEMORY = json.load(f)
            if not isinstance(LISS_MEMORY, list):
                zapis_log("Chyba při načítání paměti: Soubor neobsahuje seznam. Resetuji paměť.")
                LISS_MEMORY = []
            else:
                zapis_log(f"Načteno {len(LISS_MEMORY)} faktů z paměti.")
        except json.JSONDecodeError as e:
            zapis_log(f"Chyba při načítání paměti (JSONDecodeError): {e}. Soubor je poškozený, resetuji paměť.")
            LISS_MEMORY = []
        except Exception as e:
            zapis_log(f"Obecná chyba při načítání paměti: {e}. Resetuji paměť.")
            LISS_MEMORY = []

def uloz_fakta():
    try:
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(LISS_MEMORY, f, indent=4)
        zapis_log(f"Paměť uložena, celkem {len(LISS_MEMORY)} faktů.")
    except Exception as e:
        zapis_log(f"Chyba při ukládání paměti: {e}")

def uloz_novy_fakt(text_faktu: str):
    global LISS_MEMORY, client
    
    if not client:
        zapis_log("CHYBA: Nelze uložit fakt, Gemini klient není inicializován.")
        return False, "Gemini klient není inicializován."

    try:
        zapis_log(f"Vytvářím embedding pro nový fakt: '{text_faktu}'")
        
        embedding_vector = client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL, 
            contents=[text_faktu]
        ).embeddings[0].values 
        
        LISS_MEMORY.append({"text": text_faktu, "embedding": embedding_vector}) 
        uloz_fakta()
        
        return True, f"Uložila jsem si nový fakt: '{text_faktu}'"
        
    except Exception as e:
         zapis_log(f"CHYBA: Nelze vytvořit embedding pro nový fakt: {e}")
         return False, f"Nepodařilo se mi uložit fakt: {e}"

def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def najdi_relevantni_fakta(dotaz: str, prah_relevance=0.55) -> str:
    if not LISS_MEMORY or not client:
        return "Není k dispozici dlouhodobá paměť."

    try:
        embedding_dotazu_vector = client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=[dotaz],
        ).embeddings[0].values 
        
        np_embedding_dotazu = np.array(embedding_dotazu_vector)
        
        relevantni_fakta = []
        for fakt in LISS_MEMORY:
            if isinstance(fakt.get('embedding'), list):
                np_embedding_faktu = np.array(fakt['embedding'])
                
                podobnost = cosine_similarity(np_embedding_dotazu, np_embedding_faktu)
                
                if podobnost > prah_relevance: 
                    relevantni_fakta.append(fakt['text'])
            else:
                zapis_log(f"Varování: Poškozený fakt v paměti (chybí embedding): {fakt.get('text')}")


        if relevantni_fakta:
            zapis_log(f"Nalezena relevantní fakta (max. podobnost > {prah_relevance}).")
            return "DLOUHODOBÁ PAMĚŤ:\n" + "\n".join(relevantni_fakta)
        else:
            zapis_log("Nalezena relevantní fakta: Žádná.")
            return "Nalezena relevantní fakta: Žádná."
            
    except Exception as e:
        zapis_log(f"Chyba při vyhledávání faktů: {e}")
        return "Chyba při vyhledávání faktů."


# --- FUNKCE PRO TEXT-TO-SPEECH PŘES HTTP (beze změny) ---
def ziskat_tts_audio(text: str):
    global eleven_key
    
    if not eleven_key:
        zapis_log("TTS přeskočeno: ElevenLabs klíč není aktivní.")
        return None

    try:
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": eleven_key 
        }
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        response = requests.post(TTS_API_URL, headers=headers, json=data)
        response.raise_for_status()
        
        zapis_log("TTS audio data úspěšně získána.")
        return response.content
        
    except requests.exceptions.HTTPError as e:
        zapis_log(f"CHYBA HTTP (ElevenLabs): {e.response.status_code}. Zkontrolujte ID hlasu, klíč a limity FREE Tieru.")
    except Exception as e:
        zapis_log(f"Chyba TTS: {e}")
        return None

# --- HLAVNÍ LOGIKA CHATU (beze změny) ---
def ziskat_odpoved_liss(dotaz: str) -> str:
    global chat
    if not chat:
        zapis_log("Nelze volat API: Chat session není inicializována.")
        return "Chyba: Chat session není inicializována. Zkontrolujte log."

    relevantni_kontext = najdi_relevantni_fakta(dotaz)
    plny_dotaz = f"NÁSLEDUJE DOTAZ UŽIVATELE:\n{dotaz}\n\nKONTEXT Z DLOUHODOBÉ PAMĚTI: {relevantni_kontext}"

    try:
        response = chat.send_message(plny_dotaz)
        zapis_log("Odpověď z Gemini API úspěšně přijata.")
        return response.text
        
    except Exception as e:
        zapis_log(f"Chyba při komunikaci s Gemini: {e}")
        return f"Došlo k chybě při komunikaci s Gemini: {e}"


# --- INICIALIZAČNÍ FUNKCE (OPRAVENO PŘEPISOVÁNÍ LOGU) ---
def inicializovat_aplikaci():
    global client, chat, eleven_key

    # KROK 1: PŘEPSÁNÍ LOGU PŘI STARTU
    start_message = f"--- START APLIKACE AVA ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---"
    try:
        # Použijeme 'w' (write) pro přepsání souboru
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(start_message + '\n')
    except Exception:
        print(f"Kritická chyba: Nepodařilo se vytvořit/přepsat log soubor: {LOG_FILE}")
    print(start_message) 
        
    
    zapis_log("Zahajuji inicializaci API služeb pro Back-end...")
    
    # KROK 2: Kontrola ElevenLabs Klíče
    eleven_key = os.environ.get("ELEVEN_API_KEY")
    if eleven_key:
        zapis_log("ElevenLabs API klíč nalezen.")
    else:
        zapis_log("Upozornění: Proměnná ELEVEN_API_KEY není nastavena. Hlasový výstup nebude funkční.")

    # KROK 3: Inicializace Gemini
    try:
        if not os.environ.get("GEMINI_API_KEY"):
             zapis_log("Chyba: Proměnná GEMINI_API_KEY není nastavena. AI nebude funkční.")
             return
             
        client = genai.Client()
        nacti_pamet() 

        chat = client.chats.create(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=AVA_PERSONA
            )
        )
        zapis_log("Gemini klient a krátkodobá paměť (Ava session) inicializovány.")
        
    except Exception as e:
        zapis_log(f"KRITICKÁ CHYBA: Při inicializaci Gemini došlo k chybě: {e}")
        client = None
        chat = None
        return


# --- FLASK APLIKACE A ROUTY ---
app = Flask(__name__)
CORS(app) 

# OPRAVENÁ ROUTA: Servírování Front-endu (index.html)
@app.route('/')
def serve_index():
    # Použijeme BASE_DIR (absolutní cestu) místo os.getcwd()
    return send_from_directory(BASE_DIR, 'index.html')


# Route pro kontrolu stavu
@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "status": "OK" if chat else "CHYBA",
        "message": "Back-end je připraven." if chat else "Chyba inicializace API.",
        "model": MODEL_NAME
    })

# Route pro analýzu (Smart Save)
@app.route('/api/analyze', methods=['POST'])
def analyze():
    # ... (logika je stejná)
    def analyze_for_fact(dotaz: str) -> dict:
        if not client:
            return {"should_save": False, "extracted_fact": "AI klient není aktivní."}

        SMART_SAVE_PROMPT = f"""
        Analyzuj následující zprávu od uživatele. Rozhodni, zda zpráva obsahuje osobní,
        trvalý fakt (např. jméno, preference, povolání, koníček, vybavení jako 'volant 1080 stupňů')
        nebo jen dotaz/komentář.

        Pokud je to fakt, nastav 'should_save' na true a extrahuj stručný fakt do 'extracted_fact'.
        Pokud to není fakt hodný uložení, nastav 'should_save' na false a 'extracted_fact' na prázdný řetězec.

        Zpráva: "{dotaz}"

        Vrať odpověď POUZE jako validní JSON objekt.
        Příklad 1: {{"should_save": true, "extracted_fact": "Uživatel se jmenuje Jakub a říká mu Kubo."}}
        Příklad 2: {{"should_save": false, "extracted_fact": ""}}
        """

        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[SMART_SAVE_PROMPT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "object",
                        "properties": {
                            "should_save": {"type": "boolean"},
                            "extracted_fact": {"type": "string"},
                        }
                    }
                )
            )
            return json.loads(response.text)

        except Exception as e:
            zapis_log(f"Chyba analýzy pro Smart Save: {e}")
            return {"should_save": False, "extracted_fact": f"Chyba analýzy: {e}"}

    try:
        data = request.json
        dotaz = data.get('message', '').strip()
    except:
        return jsonify({"should_save": False, "extracted_fact": ""}), 400

    if not dotaz:
         return jsonify({"should_save": False, "extracted_fact": ""})
         
    analysis = analyze_for_fact(dotaz)
    
    return jsonify(analysis)


# Hlavní Route pro chat
@app.route('/api/chat', methods=['POST'])
def handle_chat():
    try:
        data = request.json
        dotaz = data.get('message', '').strip()
        tts_enabled = data.get('tts_enabled', False)
    except:
        return jsonify({"text": "Neplatný požadavek.", "audio_data": None}), 400

    if not dotaz:
        return jsonify({"text": "Prosím, zadejte text.", "audio_data": None})

    # 1. Kontrola příkazu /save
    if dotaz.startswith("/save "):
        fakt_k_ulozeni = dotaz[6:].strip()
        if fakt_k_ulozeni:
            success, message = uloz_novy_fakt(fakt_k_ulozeni) 
            return jsonify({"text": f"Ava (Paměť): {message}", "audio_data": None})
        else:
            return jsonify({"text": "Ava (Chyba): Musíte zadat text, který chcete uložit.", "audio_data": None})

    # 2. Získání textové odpovědi od Avy
    odpoved_text = ziskat_odpoved_liss(dotaz)
    
    audio_data = None
    if tts_enabled:
        # 3. Získání audio dat (MP3) od ElevenLabs
        audio_data = ziskat_tts_audio(odpoved_text)
        if audio_data:
            # 4. Převedení binárních dat na Base64 pro JSON přenos
            import base64
            audio_data = base64.b64encode(audio_data).decode('utf-8')
    
    return jsonify({
        "text": odpoved_text,
        "audio_data": audio_data
    })


# --- SPOUŠTĚCÍ BLOK ---
if __name__ == '__main__':
    
    try:
        import numpy as np
    except ImportError:
        print("\nCHYBA: Pro RAG je nutná knihovna 'numpy'. Nainstalujte ji: pip install numpy\n")
        exit()
        
    inicializovat_aplikaci()
    
    print("\n------------------------------------------------------------")
    print("FLASK BACK-END AVA JE SPUŠTĚN. Přístup přes: http://127.0.0.1:5000/")
    print("------------------------------------------------------------\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
