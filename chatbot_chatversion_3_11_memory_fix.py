import streamlit as st
import os
import re
import io
import altair as alt
import pydeck as pdk
import saspy
import pandas as pd
import json
from typing import List, Dict, Any
import requests
from dotenv import load_dotenv
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm

st.set_page_config(
    page_title="Infrabot",
    page_icon="🚆",     
    layout="centered",    
    initial_sidebar_state="expanded"
)


if 'sas' not in st.session_state:
    st.session_state['sas'] = None
if 'generated_query' not in st.session_state:
    st.session_state['generated_query'] = ''
if 'last_user_input' not in st.session_state:
    st.session_state['last_user_input'] = ''
if 'query_history' not in st.session_state:
    st.session_state['query_history'] = []
if 'language' not in st.session_state:
    st.session_state['language'] = 'en'  # default English
if 'validated_question' not in st.session_state:
    st.session_state['validated_question'] = ''


if 'awaiting_ambiguity' not in st.session_state:
    st.session_state['awaiting_ambiguity'] = False

if 'ambiguities_queue' not in st.session_state:
    st.session_state['ambiguities_queue'] = []  # list of {"term": "...", "options": [...]}
if 'current_ambiguity' not in st.session_state:
    st.session_state['current_ambiguity'] = None  # {"term": "...", "options": [...]}

if 'pending_details' not in st.session_state:
    st.session_state['pending_details'] = []  # list of missing details to ask for sequentially
if 'awaiting_detail' not in st.session_state:
    st.session_state['awaiting_detail'] = False  # whether we are waiting for a detail response
if 'detail_values' not in st.session_state:
    st.session_state['detail_values'] = {}  # store detail values provided by user

if 'recent_questions' not in st.session_state:
    st.session_state['recent_questions'] = []

if 'awaiting_location_question' not in st.session_state:
    st.session_state['awaiting_location_question'] = False

if 'awaiting_clarification' not in st.session_state:
    st.session_state['awaiting_clarification'] = False
if 'last_selected_locations' not in st.session_state:
    st.session_state['last_selected_locations'] = []

URL = ""
CLIENT_ID = ""
CLIENT_SECRET = ""
REFRESH_TOKEN = ""
 
url = f"{URL}/SASLogon/oauth/token"
 
headers = {
    "Accept": "application/json",
    "Content-Type": "application/x-www-form-urlencoded"
}
 
data = {
    "grant_type": "refresh_token",
    "refresh_token": REFRESH_TOKEN
}
 
# Make request with basic auth
response = requests.post(
    url,
    headers=headers,
    data=data,
    auth=(CLIENT_ID, CLIENT_SECRET),  
    verify=False  
)
 
access_token = response.json()["access_token"]

# -----------------------------------------------------------------------------
# Language definitions and UI translations
# -----------------------------------------------------------------------------
languages: Dict[str, str] = {
    'en': 'English',
    'fr': 'Français',
    'nl': 'Nederlands'
}

UI_TEXT: Dict[str, Dict[str, str]] = {
    'enter_question': {
        'en': "🔍 Enter your question:",
        'fr': "🔍 Entrez votre question :",
        'nl': "🔍 Voer uw vraag in:"
    },
    'run_query': {
        'en': "▶️ Run Query in SAS",
        'fr': "▶️ Exécuter la requête dans SAS",
        'nl': "▶️ Voer query uit in SAS"
    },
    'query_history': {
        'en': "🕘 Question History",
        'fr': "🕘 Historique des questions",
        'nl': "🕘 Vragenhistoriek"
    },
    'clear_history': {
        'en': "🧹 Clear History",
        'fr': "🧹 Effacer l'historique",
        'nl': "🧹 Geschiedenis wissen"
    },
    'no_history': {
        'en': "No previous queries yet.",
        'fr': "Aucune requête précédente.",
        'nl': "Nog geen eerdere queries."
    },
    'enter_question_tip': {
        'en': "Enter a question above and click **Generate Query** to start.",
        'fr': "Saisissez une question ci-dessus et cliquez sur **Générer la requête** pour commencer.",
        'nl': "Voer hierboven een vraag in en klik op **Genereer query** om te beginnen."
    },
    'sas_log_output': {
        'en': "📄 Show SAS Log Output",
        'fr': "📄 Afficher le journal SAS",
        'nl': "📄 Toon SAS-logboek"
    },
    'download_pdf': {
    'en': "⬇️ Download PDF",
    'fr': "⬇️ Télécharger PDF",
    'nl': "⬇️ PDF downloaden"
    },
    'ask_train_type': {
        'en': "Please specify the train type.",
        'fr': "Veuillez préciser le type de train.",
        'nl': "Gelieve het treintype te specificeren."
    },
    'ask_date': {
        'en': "Please specify the date.",
        'fr': "Veuillez préciser la date.",
        'nl': "Gelieve de datum te specificeren."
    },
    'location_button': {
        'en': "📍 Locations",
        'fr': "📍 Lieux",
        'nl': "📍 Locaties"
    },
    'select_locations': {
        'en': "Select PTCAR(s) to show on map:",
        'fr': "Sélectionnez des PTCAR(s) à afficher sur la carte :",
        'nl': "Selecteer PTCAR(s) om op de kaart te tonen:"
    },
    'show_locations': {
        'en': "Show location(s)",
        'fr': "Afficher emplacement(s)",
        'nl': "Toon locatie(s)"
    },
    'show_code': {
        'en': "Show generated code",
        'fr': "Afficher le code généré",
        'nl': "Toon gegenereerde code"
    },
    'ask_followup_locations': {
        'en': "Would you like to ask a question about these locations? For example:*What was the number of trains passing through those places in June 2023?*",
        'fr': "Souhaitez-vous poser une question concernant ces lieux ? Par exemple :*Quel était le nombre de trains passant par ces lieux en juin 2023 ?*",
        'nl': "Wil je een vraag stellen over deze locaties? Bijvoorbeeld:*Wat was het aantal treinen dat in juni 2023 door deze plaatsen reed?*"
    },
    'generated_executing': {
    'en': "I generated the code, now I need to execute it…",
    'fr': "J'ai généré le code, maintenant je dois l'exécuter…",
    'nl': "Ik heb de code gegenereerd, nu moet ik die uitvoeren…"
    },
    'visualize_result': {'en': "📈 Visualize this result", 'fr': "📈 Visualiser ce résultat", 'nl': "📈 Dit resultaat visualiseren"},
    
    'sas_status': {'en': "SAS status", 'fr': "Statut SAS", 'nl': "SAS-status"},
    'connected': {'en': "Connected", 'fr': "Connecté", 'nl': "Verbonden"},
    'disconnected': {'en': "Disconnected", 'fr': "Déconnecté", 'nl': "Verbroken"},

    'sas_connect_ok': {'en': "✅ Connected to SAS", 'fr': "✅ Connecté à SAS", 'nl': "✅ Verbonden met SAS"},
    'sas_connect_fail': {'en': "❌ SAS connection failed", 'fr': "❌ Échec de la connexion SAS", 'nl': "❌ SAS-verbinding mislukt"},
    'query_may_take': {'en': "⚠️ Query involves large tables... this may take a moment.",
                       'fr': "⚠️ La requête implique de grandes tables… cela peut prendre un moment.",
                       'nl': "⚠️ De query gebruikt grote tabellen… dit kan even duren."},
    'running_query': {'en': "Running query in SAS...", 'fr': "Exécution de la requête dans SAS…", 'nl': "Query uitvoeren in SAS…"},
    'executing_query': {'en': "Executing query in SAS...", 'fr': "Exécution de la requête dans SAS…", 'nl': "Query uitvoeren in SAS…"},
    'regenerating_sql': {'en': "Regenerating SQL query after SAS error...",
                         'fr': "Régénération de la requête SQL après une erreur SAS…",
                         'nl': "SQL opnieuw genereren na SAS-fout…"},
    'sas_error_detected': {'en': "SAS error detected", 'fr': "Erreur SAS détectée", 'nl': "SAS-fout gedetecteerd"},
    'no_data_returned': {'en': "No data returned, possibly an incorrect query",
                         'fr': "Aucune donnée renvoyée, requête probablement incorrecte",
                         'nl': "Geen gegevens teruggegeven, mogelijk onjuiste query"},

    'ambiguous_station_msg': {'en': "The station name is ambiguous. Please select from the options below.",
                              'fr': "Le nom de la gare est ambigu. Veuillez choisir parmi les options ci-dessous.",
                              'nl': "De stationsnaam is dubbelzinnig. Kies een optie hieronder."},
    'choose_correct_station': {'en': "Please choose the correct station for",
                               'fr': "Veuillez choisir la gare correcte pour",
                               'nl': "Kies het juiste station voor"},
    'confirm_selection': {'en': "Confirm selection", 'fr': "Confirmer la sélection", 'nl': "Selectie bevestigen"},
    'next_ambiguous': {'en': "Next ambiguous station:", 'fr': "Gare ambiguë suivante :", 'nl': "Volgend dubbelzinnig station:"},
    'choose_one_of': {'en': "That doesn't match any option. Please choose one of:",
                      'fr': "Cela ne correspond à aucune option. Veuillez choisir parmi :",
                      'nl': "Dat komt met geen enkele optie overeen. Kies één van:"},
    'validating_question': {'en': "Validating your question...", 'fr': "Validation de votre question…", 'nl': "Je vraag valideren…"},
    'generating_sql': {'en': "Generating SQL query...", 'fr': "Génération de la requête SQL…", 'nl': "SQL-query genereren…"},

    'email_generate_section': {
        'en': "✉️ Generate email template",
        'fr': "✉️ Générer un modèle d'e-mail",
        'nl': "✉️ E-mailsjabloon genereren"
    },
    'email_generate_btn': {
        'en': "Generate email template",
        'fr': "Générer le modèle d'e-mail",
        'nl': "E-mailsjabloon genereren"
    },
    'email_generating': {
        'en': "Generating email template...",
        'fr': "Génération du modèle d'e-mail…",
        'nl': "E-mailsjabloon genereren…"
    },
    'email_regenerate_btn': {
        'en': "Regenerate",
        'fr': "Régénérer",
        'nl': "Opnieuw genereren"
    },
    'email_regenerating': {
        'en': "Regenerating email template...",
        'fr': "Régénération du modèle d'e-mail…",
        'nl': "E-mailsjabloon opnieuw genereren…"
    },

}

# -----------------------------------------------------------------------------
# Gemini model configuration
# -----------------------------------------------------------------------------
@st.cache_resource
def get_gemini_model(api_key: str):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.0-flash')

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
model = get_gemini_model(GEMINI_API_KEY)

# -----------------------------------------------------------------------------
# Load schema and station list
# -----------------------------------------------------------------------------
@st.cache_data
def load_schema() -> str:
    schema: List[str] = []
    data_folder = 'data'
    excel_files = ['DESCRIPTION_FIELDS_v2.xlsx']
    for file in excel_files:
        file_path = os.path.join(data_folder, file)
        if os.path.exists(file_path):
            try:
                df = pd.read_excel(file_path, sheet_name='Sheet1', header=None)
                for _, row in df.iterrows():
                    field = row[1] if len(row) > 1 else ''
                    description = row[2] if len(row) > 2 else ''
                    column_info = row[3] if len(row) > 3 else ''
                    if field and description:
                        if column_info:
                            schema.append(f"{field}: {description} (Table: {column_info})")
                        else:
                            schema.append(f"{field}: {description}")
            except Exception as e:
                st.error(f"Error reading {file}: {str(e)}")
    return "\n".join(schema) if schema else "No schema found."

@st.cache_data
def load_places() -> List[str]:
    with open('data/ptcars.txt', 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]

FIELD_CONTEXT: str = load_schema()
places: List[str] = load_places()

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def is_sas_connected() -> bool:
    return st.session_state.get('sas') is not None

def sas_status_indicator() -> None:
    if is_sas_connected():
        color = "green"
        status = t('connected')
    else:
        color = "red"
        status = t('disconnected')
    align = "text-align:center;"
    st.markdown(
        f"""
        <p style="font-weight:bold; margin:-3px 0; {align}">
            {t('sas_status')}: 
            <span style="color:{color}; font-weight:bold;">●</span> {status}
        </p>
        """,
        unsafe_allow_html=True
    )

def t(key: str) -> str:
    lang = st.session_state['language']
    return UI_TEXT.get(key, {}).get(lang, key)

LOG_FILE = "chatbot_log.txt"

def append_to_logfile(entry: str):
    """Append a timestamped entry to the chatbot log file."""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] {entry}\n")
    except Exception as e:
        st.error(f"Failed to write log: {e}")

# -----------------------------------------------------------------------------
# Gemini helper functions for multi‑step processing
# -----------------------------------------------------------------------------
def call_gemini(prompt: str) -> str:
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        if '```' in response_text:
            response_text = re.sub(r"```\w*", '', response_text).replace('```', '').strip()
        return response_text
    except Exception as e:
        return json.dumps({'error': f'Error calling model: {str(e)}'})

# -----------------------------------------------------------------------------
# Conversation memory helpers
# -----------------------------------------------------------------------------
def _get_text_conversation_history(max_messages: int = 8) -> List[Dict[str, str]]:
    """Return the last few text-only chat messages (role/content).

    Skips structured assistant payloads like dataframes, code, logs, and maps.
    """
    history: List[Dict[str, str]] = []
    try:
        for m in st.session_state.get('messages', []):
            role = m.get('role')
            content = m.get('content')
            if role in ('user', 'assistant') and isinstance(content, str) and content.strip():
                history.append({'role': role, 'content': content.strip()})
    except Exception:
        return []
    return history[-max_messages:]


def rewrite_with_history(user_input: str, language: str) -> str:
    """Use recent conversation to rewrite a follow-up into a standalone question.

    - Keeps the original language.
    - If no rewrite is needed or model fails, returns the original input.
    """
    try:
        hist = _get_text_conversation_history(max_messages=8)
        if not hist:
            return user_input

        hist_lines = []
        for h in hist:
            role = 'User' if h['role'] == 'user' else 'Assistant'
            # Trim very long lines to keep prompt compact
            text = h['content']
            if len(text) > 500:
                text = text[:500] + '...'
            hist_lines.append(f"{role}: {text}")
        hist_text = "\n".join(hist_lines)

        prompt = f"""
You help rewrite user inputs into standalone questions about train operations in Belgium.
Use the conversation history to resolve references (e.g., "that", "those", "same", pronouns, implicit dates/stations).
Maintain the user's language: {language}. If the message is already standalone, keep it unchanged.

Conversation history (most recent last):
{hist_text}

New message: "{user_input}"

Return ONLY the rewritten message text with no quotes or extra commentary.
"""
        rewritten = call_gemini(prompt).strip()
        if not rewritten or rewritten.lower().startswith('{"error"'):
            return user_input
        return rewritten
    except Exception:
        return user_input

# -----------------------------------------------------------------------------
# Utility functions for ambiguity and clarification parsing
# -----------------------------------------------------------------------------

def _extract_ambiguous_term(options: List[str]) -> str:
    if not options:
        return ''
    # Normalise to uppercase for comparison
    opts = [opt.upper() for opt in options]
    s1 = min(opts)
    s2 = max(opts)
    prefix = ''
    for i, ch in enumerate(s1):
        if i >= len(s2) or s2[i] != ch:
            prefix = s1[:i]
            break
    else:
        prefix = s1
    return prefix.rstrip('- ').strip()

def _normalize_ambiguities(amb):
    if not amb:
        return []
    # New format already OK
    if isinstance(amb, list) and amb and isinstance(amb[0], dict) and 'options' in amb[0]:
        # ensure uppercase options
        norm = []
        for item in amb:
            term = (item.get('term') or '').strip()
            opts = [o.upper() for o in item.get('options', [])]
            if term == '':
                term = _extract_ambiguous_term(opts) or 'STATION'
            norm.append({'term': term, 'options': opts})
        return norm
    if isinstance(amb, list) and amb and isinstance(amb[0], str):
        opts = [o.upper() for o in amb]
        return [{'term': _extract_ambiguous_term(opts) or 'STATION', 'options': opts}]
    return []

def enrich_question_with_locations(question: str, locations: List[str]) -> str:
    if not locations:
        return question

    loc_list = ", ".join([loc.upper() for loc in locations])

    patterns = [
        r'\b(those|these)\s+(places|locations|ptcars)\b',
        r'\b(diese|ces|dieze?)\s+(lieux|plaatsen|locaties)\b',
    ]
    replaced = question
    for pat in patterns:
        replaced_new = re.sub(pat, loc_list, replaced, flags=re.IGNORECASE)
        if replaced_new != replaced:
            replaced = replaced_new
            break

    if replaced == question:
        if replaced.strip().endswith('?'):
            replaced = replaced[:-1].strip() + f" for {loc_list}?"
        else:
            replaced = replaced.strip() + f" for {loc_list}"
    return replaced

def parse_missing_details(clarification_msg: str) -> List[str]:
    msg = clarification_msg.lower() if clarification_msg else ''
    missing: List[str] = []
    type_keywords = [
        'train type', 'type de train', 'treintype', 'type du train', 'typo de tren', 'tipo de tren'
    ]
    date_keywords = [
        'date', 'datum', 'fecha', 'data'
    ]
    for kw in type_keywords:
        if kw in msg and 'type' not in missing:
            missing.append('type')
            break
    for kw in date_keywords:
        if kw in msg and 'date' not in missing:
            missing.append('date')
            break
    return missing

def _detect_follow_up(message: str) -> bool:
    if not message:
        return False
    lower = message.lower()
    if any(token in lower for token in ['this', 'same', 'previous', 'repeat']) and re.search(r"\b20\d{2}\b", lower):
        return True
    if 'instead of' in lower and re.search(r"\b20\d{2}\b", lower):
        return True
    return False


def _apply_follow_up_to_question(message: str, previous_question: str) -> str:
    if not previous_question:
        return message
    # find all 4‑digit years in the follow‑up message
    new_years = re.findall(r"\b(20\d{2})\b", message)
    if new_years:
        # look for explicit 'instead of' old year
        old_years = re.findall(r"(?<=instead of )\s*(20\d{2})", message, flags=re.IGNORECASE)
        target = previous_question
        # Use first match for new year
        new_year = new_years[0]
        if old_years:
            old_year = old_years[0]
            # Replace all occurrences of the old year with the new year
            modified = re.sub(old_year, new_year, target)
            return modified
        else:
            # No explicit old year; replace first year found in previous question
            m = re.search(r"\b(20\d{2})\b", target)
            if m:
                old_year = m.group(1)
                modified = re.sub(old_year, new_year, target, count=1)
                return modified
            else:
                # If no year in previous question, append the new year with a preposition
                target = target.rstrip('?')
                return f"{target} in {new_year}?" if '?' in previous_question else f"{target} in {new_year}"
    # If there is no year in the message, treat the follow‑up as an append
    return previous_question.rstrip('?') + ' ' + message.strip() + ('?' if '?' in previous_question else '')

def validate_question_llm(question: str) -> Dict[str, Any]:
    stations_list = ', '.join([p.upper() for p in places])

    prompt = f"""
You are a helpful assistant for validating user questions about train operations in Belgium.  The user may ask in French, English or Dutch.  Always respond in the same language that the question was asked.

You are provided with a list of valid station names (ptcars) in Belgium: {stations_list}.

You are also provided with a schema of available fields and their tables in the underlying dataset:
{FIELD_CONTEXT}

You are provided with a list of sample questions:
1. how many hkv trains passed through boechout in june 2023
2. what are the ptcars between wondelgem and gent-dampoort on line 58?
3. show the summary of numner of trains passing through boechout in 2023
4. what was the average delay of trains passing through leuven in june 2023

Your tasks:
1. Determine whether the question is about trains in Belgium.  If not, set the 'error' field explaining that only questions about trains in Belgium are supported.
2. Check station names. Correct misspellings to the closest name from the provided list. 
   If a name matches multiple stations, include an 'ambiguities' field as an array of objects:
   [{{ "term": "<the ambiguous text as it appears in the user question>", "options": ["NAME A", "NAME B", ...] }}, ...]
   Do not choose on the user's behalf.

   If you cannot identify the exact 'term' substring, use the longest common prefix across options as the 'term'.
3. Based on question type different information should be provided. Compare user's question with list of sample questions and determine whether some important details should be further clarified by user. If any of these details are missing, set the 'clarification' field asking the user to provide the missing information.
4. If the question asks for information that is not available in the provided schema (for example, the colour of train wagons), include a 'clarification' field explaining that this information is not available and cannot be included in the query.
5. If no clarification is needed, return the rewritten question (with corrected station names) in the 'validated_question' field.
6. If any other data is needed to write a query ask about it in 'clarification' field.
7. You only have train data for years 2023 and 2024. If another year was specified in the question, explain that you cannot answer it in the language of the question.

Return your answer strictly as a JSON object with up to four fields: 'validated_question', 'clarification', 'ambiguities' and 'error'.  Omit any fields that are not needed.  Do not wrap the JSON in code fences.

User question: "{question}"
"""
    response_text = call_gemini(prompt)
    try:
        result = json.loads(response_text)
        allowed_keys = {"validated_question", "clarification", "ambiguities", "error"}
        return {k: v for k, v in result.items() if k in allowed_keys}
    except Exception:
        return {"error": f"Invalid response from model: {response_text}"}

def generate_query_single_step(validated_question: str, error: str = None) -> Dict[str, Any]:
    examples = """
- Example: For "how many hkv trains passed through boechout in 2023 in every month", return:
PROC SQL;
SELECT MONTH(t.DATDEP) AS month,
       COUNT(DISTINCT t.TRAIN_ID) AS num_hkv_trains
FROM infrapq.train_datdep_2023 AS t
INNER JOIN infrapq.train_ptcar_2023 AS p
  ON t.TRAIN_ID = p.TRAIN_ID AND t.DATDEP = p.DATDEP
INNER JOIN infrapq.current_ptcar_bel AS c
  ON p.PTCAR_NO = c.PTCAR_NO
WHERE t.TRAINTYPPAV = 'HKV'
  AND c.PTCAR_LG_NM_NL = "BOECHOUT"
GROUP BY month;
QUIT;

- Example: For "what are the ptcars between wondelgem and gent-dampoort on line 58?", return:
PROC SQL;
SELECT DISTINCT p.PTCAR_LG_NM_NL
FROM infrapq.current_ptcar_by_line_bel AS p
INNER JOIN infrapq.current_line_bel AS l ON p.LINE_ID = l.LINE_ID
WHERE l.LINE_NO = '58'
AND p.LINE_DIST BETWEEN
  (SELECT MIN(LINE_DIST) FROM infrapq.current_ptcar_by_line_bel WHERE PTCAR_LG_NM_NL = 'WONDELGEM' AND LINE_NO = '58')
AND
  (SELECT MAX(LINE_DIST) FROM infrapq.current_ptcar_by_line_bel WHERE PTCAR_LG_NM_NL = 'GENT-DAMPOORT' AND LINE_NO = '58');
QUIT;

- Example: For "How many trains were in Belgium in 2023 and 2024?", return:
proc sql;
    select '2023' as year, count(distinct train_id) as count
    from infrapq.train_ptcar_2023
    union
    select '2024' as year, count(distinct train_id) as count
    from infrapq.train_ptcar_2024;
quit;
"""

    parts = []
    parts.append("You generate SAS PROC SQL for train operations in Belgium.\n")
    parts.append("Context (schema/fields):\n")
    parts.append(str(FIELD_CONTEXT) + "\n")
    parts.append(
        "Rules:\n"
        "- Tables available: infrapq.train_datdep_&YEAR, infrapq.train_ptcar_&YEAR, infrapq.train_comp_&YEAR,\n"
        "  infrapq.datdepcalendar, infrapq.current_ptcar_bel, infrapq.current_ptcar,\n"
        "  infrapq.current_linesec_bel, infrapq.current_line_bel,\n"
        "  infrapq.current_ptcar_by_line_bel, infrapq.current_ptcar_by_linesec_bel, infrapq.train_linesec_&YEAR.\n"
        "  Assume data exists for 2023 and 2024 unless the user asks otherwise.\n"
        "- Do NOT CREATE TABLE. Return a SELECT only; end with QUIT;.\n"
        "- Use correct SAS date literals like '14AUG2023'd.\n"
        "- When asked for top n results, use PROC SQL outobs=n;.\n"
        "- When using COUNT, assign an alias (e.g., COUNT(DISTINCT TRAIN_ID) AS count).\n"
        "- Write station names in CAPITALS in the query.\n"
        "- Write tables names in lower case in the query.\n"
        "- Character columns are DAY_OF_WEEK, DIRECTION, HOLIDAY, TRAIN_GRP, ARR_BEL and LINE_NO.\n"
        "- WEEKEND is a character column, place its values in commas.\n"
        "- Prefer equi-joins and indexed columns where reasonable.\n"
        "- If month breakdown is requested, SELECT MONTH(DATDEP) AS month and GROUP BY it.\n"
    )
    if error:
        parts.append("- Previous SAS error log to fix:\n" + str(error) + "\n")

    parts.append("Examples to imitate:\n" + examples + "\n")
    parts.append('Validated question: "' + str(validated_question) + '"\n')
    parts.append("Return ONLY the SQL text (PROC SQL...QUIT;). No JSON, no prose.")

    prompt = "\n".join(parts)

    sql = call_gemini(prompt).strip()
    return {"query": sql}

def _execute_current_query_now():
    """
    Connects to SAS if needed and executes st.session_state['generated_query'] immediately.
    Mirrors the old button handler’s behavior, including auto-regeneration on SAS error.
    """
    with st.spinner("Connecting to SAS and executing query..."):
        try:
            # Ensure SAS session
            if st.session_state['sas'] is None:
                try:
                    st.session_state['sas'] = saspy.SASsession(cfgname='httpsviya', authtoken=access_token)
                    st.success(t('sas_connect_ok'))
                except Exception as e:
                    st.session_state['sas'] = None
                    st.error(f"{t('sas_connect_fail')}: {e}")
                    st.session_state['run_query_finished'] = True
                    return

            validated_question = st.session_state.get('validated_question', '')
            current_query = st.session_state['generated_query']

            df, error_msg, log = run_sas_query(current_query)

            if error_msg:
                with st.spinner(t('regenerating_sql')):
                    query_result = generate_query_single_step(validated_question, error=log)
                new_query = (query_result.get('query') or '').strip()
                st.session_state['generated_query'] = new_query
                st.session_state['messages'].append({'role': 'assistant', 'content': {'type': 'code', 'data': new_query}})

                df, error_msg, log = run_sas_query(new_query)
                if error_msg:
                    st.session_state['messages'].append({'role': 'assistant', 'content': error_msg})
                    st.session_state['messages'].append({'role': 'assistant', 'content': {'type': 'log', 'data': log}})
                    append_to_logfile(f"SAS LOG:\n{log}")
                    st.session_state['run_query_finished'] = True
                    return

            if df is not None and not error_msg:
                st.session_state['messages'].append({'role': 'assistant', 'content': {'type': 'dataframe', 'data': df}})
                if 'ERROR:' in (log or '').upper():
                    st.session_state['messages'].append({'role': 'assistant', 'content': {'type': 'log', 'data': log}})
                    append_to_logfile(f"SAS LOG:\n{log}")
                st.session_state['query_history'].append({
                    'question': st.session_state.get('validated_question') or st.session_state.get('last_user_input') or '',
                    'data': df.copy()
                })
            st.session_state['run_query_finished'] = True
        except Exception as e:
            st.session_state['messages'].append({'role': 'assistant', 'content': f"❌ Error during SAS execution: {str(e)}"})
            st.session_state['run_query_finished'] = True

def run_sas_query(query: str):
    try:
        if "train_datdep" in query or "train_ptcar" in query:
            st.warning(t('query_may_take'))
        st.info(t('running_query'))

        sas = st.session_state['sas']

        code = f"""
ods listing close;
ods html5 (id=web) body=stdout;
options nocenter;
{query}
ods html5 (id=web) close;
ods listing;
"""
        with st.spinner(t('executing_query')):
            sas_result = sas.submit(code, results='HTML')
            log = sas_result.get('LOG', '')
            if 'ERROR:' in log.upper():
                return None, t('sas_error_detected'), log

            html = sas_result.get('LST', '')
            # Use safer HTML parsers to avoid potential lxml/libxml2 heap corruption
            # seen as "malloc(): unsorted double linked list corrupted" on some systems.
            tables = []
            if html:
                try:
                    # Prefer BeautifulSoup-based parser (pure Python)
                    tables = pd.read_html(io.StringIO(html), flavor='bs4')
                except Exception:
                    try:
                        # Fallback to html5lib which is also safer than lxml for malformed HTML
                        tables = pd.read_html(io.StringIO(html), flavor='html5lib')
                    except Exception:
                        tables = []
            # pick the last non-empty table
            df = next((t for t in reversed(tables) if t.shape[0] > 0 and t.shape[1] > 0), None)
            if df is None:
                return None, t('no_data_returned'), log

            # Drop SAS' "Obs" index column if it shows up
            if 'Obs' in df.columns:
                try:
                    if (df['Obs'].astype(int).diff().dropna() == 1).all():
                        df = df.drop(columns=['Obs'])
                except Exception:
                    pass
            return df, None, log
    except Exception as e:
        return None, f"An error occurred: {str(e)}", ''
    
def render_dataframe_with_viz(df, key_prefix="df"):
    st.dataframe(df)

    # Auto-detect columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    timeish_cols = [c for c in df.columns
                    if pd.api.types.is_datetime64_any_dtype(df[c]) or re.search(r"(date|time|month|year|period)", c, re.I)]

    if not numeric_cols:
        return  # nothing to plot

    with st.expander(t('visualize_result')):
        chart_choice = st.radio(
            "Chart",
            ["Bar", "Time series", "Histogram"],
            horizontal=True,
            key=f"{key_prefix}_chart"
        )

        if chart_choice == "Histogram":
            col = st.selectbox("Column (numeric)", numeric_cols, key=f"{key_prefix}_histcol")
            bins = st.slider("Bins", 5, 60, 20, key=f"{key_prefix}_bins")
            chart = (alt.Chart(df)
                       .mark_bar()
                       .encode(x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=bins)),
                               y="count()",
                               tooltip=list(df.columns)))
            st.altair_chart(chart.properties(height=300), use_container_width=True)
        else:
            x_candidates = timeish_cols or df.columns.tolist()
            # sensible defaults: date-ish for X, first numeric for Y
            default_x = next((c for c in ["DATDEP", "date", "month", "period"] if c in df.columns), x_candidates[0])
            x = st.selectbox("X-axis", x_candidates, index=x_candidates.index(default_x) if default_x in x_candidates else 0,
                             key=f"{key_prefix}_x")
            y = st.selectbox("Y-axis (numeric)", numeric_cols, key=f"{key_prefix}_y")

            dfp = df.copy()
            x_field = x

            # If X is a month number, coerce to a dummy date so time axis sorts correctly
            if x.lower() == "month" and not pd.api.types.is_datetime64_any_dtype(dfp[x]):
                with pd.option_context('mode.chained_assignment', None):
                    try:
                        dfp["__x__"] = pd.to_datetime(dfp[x].astype(int), format="%m", errors="coerce")
                        x_field = "__x__"
                    except Exception:
                        x_field = x  # fall back

            is_time = pd.api.types.is_datetime64_any_dtype(dfp[x_field]) or x_field in timeish_cols
            base = alt.Chart(dfp)
            mark = base.mark_line(point=True) if chart_choice == "Time series" else base.mark_bar()

            chart = mark.encode(
                x=alt.X(f"{x_field}:{'T' if is_time else 'O'}"),
                y=alt.Y(f"{y}:Q"),
                tooltip=list(dfp.columns)
            )
            st.altair_chart(chart.properties(height=300), use_container_width=True)

    with st.expander(t('email_generate_section'), expanded=False):
        # Keys to store generation state
        gen_key = f"{key_prefix}_email_generated"
        txt_key = f"{key_prefix}_email_text"

        if not st.session_state.get(gen_key, False):
            # One-shot trigger to generate; shows spinner while LLM runs
            if st.button(t('email_generate_btn'), key=f"{key_prefix}_email_btn"):
                with st.spinner(t('email_generating')):
                    question = st.session_state.get('validated_question') or st.session_state.get('last_user_input') or ''
                    lang = st.session_state.get('language', 'en')
                    try:
                        email_text = generate_email_template(question, df, language=lang)
                    except Exception as e:
                        email_text = f"(Failed to generate email: {e})"
                    st.session_state[txt_key] = email_text
                    st.session_state[gen_key] = True
                    st.rerun()

        # If already generated, show the copy-ready block
        if st.session_state.get(gen_key, False):
            st.code(st.session_state.get(txt_key, ""), language="markdown")
            # Optional: small reset button if you want to regenerate
            if st.button(t('email_regenerate_btn'), key=f"{key_prefix}_email_regen"):
                with st.spinner(t('email_regenerating')):
                    question = st.session_state.get('validated_question') or st.session_state.get('last_user_input') or ''
                    lang = st.session_state.get('language', 'en')
                    try:
                        email_text = generate_email_template(question, df, language=lang)
                    except Exception as e:
                        email_text = f"(Failed to generate email: {e})"
                    st.session_state[txt_key] = email_text
                    st.session_state[gen_key] = True
                    st.rerun()

def _summarize_df_for_email(df: pd.DataFrame, max_rows: int = 3) -> str:
    """
    Produce a compact textual summary of the dataframe that the LLM can use.
    Keeps it tiny to control token use.
    """
    lines = []
    # shape
    lines.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    # column list
    cols_preview = ", ".join(map(str, df.columns.tolist()[:10]))
    lines.append(f"Columns (first 10): {cols_preview}")
    # quick stats for numeric columns
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        desc = df[num_cols].describe().round(3).to_dict()
        # Only top-level stats for 1-2 columns to keep it concise
        for c in num_cols[:2]:
            stats = desc.get(c, {})
            if stats:
                parts = []
                for k in ["count", "mean", "sum", "min", "max"]:
                    if k in stats:
                        parts.append(f"{k}={stats[k]}")
                if parts:
                    lines.append(f"Stats for {c}: " + ", ".join(parts))
    # tiny head preview
    head = df.head(max_rows)
    lines.append("Top rows preview:")
    lines.append(head.to_csv(index=False).strip())
    return "\n".join(lines)


def generate_email_template(question: str, df: pd.DataFrame, language: str = 'en') -> str:

    df_brief = _summarize_df_for_email(df)
    lang = language or 'en'

    # Simple localized greetings/sign-off (extend as needed)
    GREET = {
        'en': "Hi,",
        'fr': "Bonjour,",
        'nl': "Hoi,"
    }.get(lang, "Hi,")
    SIGN  = {
        'en': "Best regards,",
        'fr': "Cordialement,",
        'nl': "Met vriendelijke groeten,"
    }.get(lang, "Best regards,")

    prompt = f"""
You write short, professional emails.

Language: {lang}
Constraints:
- Be concise (100-150 words).
- Restate the initial question once.
- Give the answer based on the provided result summary.
- Include 2-4 bullet points with key numbers or findings (if applicable).
- Avoid code blocks. No Markdown headings. Plain email text only.
- Neutral, professional tone. No exaggerated claims.

Initial question:
{question}

Result summary (for your eyes to extract facts; do NOT paste this verbatim):
{df_brief}

Use this structure:
{GREET}

One-paragraph answer with the conclusion and context.

• bullet
• bullet
• bullet (optional)

Next steps (one brief sentence).

{SIGN}
Infrabot
"""

    email_text = call_gemini(prompt)
    return email_text.strip()

@st.cache_data(show_spinner=False)
def summarize_query(code: str, language: str = 'en') -> str:

    lang = language or 'en'

    prompt = f"""
Extract from the code which year was used to get the results. If no year was needed also explain it. 
Communicate it to the user in form: I used YEAR to retrieve the results.

Language: {lang}

Code:
{code}

"""

    email_text = call_gemini(prompt)
    return email_text.strip()

def _dataframe_to_pdf_bytes(question: str, df: pd.DataFrame) -> bytes:

    df_str = df.astype(str)

    data = [list(df_str.columns)] + df_str.values.tolist()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()
    title = Paragraph(question or "Result", styles["Heading2"])

    # Compute column widths so the table fits the page
    page_width = A4[0] - (15 * mm + 15 * mm)
    ncols = max(1, len(df_str.columns))
    # Simple equal-width distribution; ReportLab will wrap text as needed
    col_width = page_width / ncols
    col_widths = [col_width] * ncols

    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fbfbfb")]),
    ]))

    story = [title, Spacer(1, 6 * mm), table]
    doc.build(story)
    buffer.seek(0)
    return buffer.read()

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
def main() -> None:

    with st.sidebar:
        st.sidebar.image("data/infrabel.png", use_container_width=True)
        sas_status_indicator()
        st.sidebar.markdown("---")

    selected_lang = st.sidebar.selectbox(
        "Select language / Sélectionnez la langue / Selecteer taal",
        options=list(languages.keys()),
        format_func=lambda x: languages[x],
        index=list(languages.keys()).index(st.session_state['language'])
    )
    st.session_state['language'] = selected_lang

    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'run_query_finished' not in st.session_state:
        st.session_state['run_query_finished'] = False

    chat_input_key = 'chat_input'

    if 'location_mode' not in st.session_state:
        st.session_state['location_mode'] = False
    if st.sidebar.button(t('location_button'), key='location_button_sidebar'):
        st.session_state['location_mode'] = not st.session_state['location_mode']
    if st.session_state.get('location_mode', False):
        selected_locations = st.sidebar.multiselect(t('select_locations'), options=sorted(places), key='location_multiselect_sidebar')
        if selected_locations:
            if st.sidebar.button(t('show_locations'), key='show_locations_sidebar'):
                conditions = " OR ".join([f"PTCAR_LG_NM_NL = '{loc.upper()}'" for loc in selected_locations])
                query_text = (
                    "PROC SQL;\n"
                    "SELECT PTCAR_LG_NM_NL, "
                    "PUT(PTCAR_GPSLAT, BEST32.) AS latitude, "
                    "PUT(PTCAR_GPSLONG, BEST32.) AS longitude\n"
                    "FROM infrapq.current_ptcar\n"
                    f"WHERE {conditions};\n"
                    "QUIT;"
                )
                st.session_state['messages'].append({
                    'role': 'assistant',
                    'content': {'type': 'code', 'data': query_text}
                })
                if st.session_state['sas'] is None:
                    try:
                        st.session_state['sas'] = saspy.SASsession(cfgname='httpsviya', authtoken=access_token)
                        st.success(t('sas_connect_ok'))
                    except Exception as e:
                        st.session_state['sas'] = None
                        st.session_state['messages'].append({'role': 'assistant', 'content': f"{t('sas_connect_fail')}: {e}"})
                        st.session_state['location_mode'] = False
                        st.rerun()
                df, error_msg, log = run_sas_query(query_text)
                if error_msg:
                    st.session_state['messages'].append({'role': 'assistant', 'content': error_msg})
                    st.session_state['messages'].append({'role': 'assistant', 'content': {'type': 'log', 'data': log}})
                    append_to_logfile(f"SAS LOG:\n{log}")
                if df is not None:
                    try:
                        df_map = df.copy()
                        df_map.columns = [col.lower() for col in df_map.columns]
                        if {'ptcar_lg_nm_nl', 'latitude', 'longitude'}.issubset(df_map.columns):
                            df_map['latitude'] = df_map['latitude'].astype(float)
                            df_map['longitude'] = df_map['longitude'].astype(float)
                            map_data = pd.DataFrame({
                                'lat': df_map['latitude'],
                                'lon': df_map['longitude'],
                                'name': df_map['ptcar_lg_nm_nl']
                            })
                            st.session_state['messages'].append({'role': 'assistant', 'content': {'type': 'map', 'data': map_data}})
                    except Exception as map_err:
                        st.session_state['messages'].append({'role': 'assistant', 'content': f"Error rendering map: {str(map_err)}"})
                
                st.session_state['last_selected_locations'] = [loc.upper() for loc in selected_locations]
                st.session_state['awaiting_location_question'] = True
                # st.session_state['messages'].append({
                #     'role': 'assistant',
                #     'content': t('ask_followup_locations')
                # })
                st.session_state['location_mode'] = False
                st.rerun()

    for i, msg in enumerate(st.session_state['messages']):
        role = msg.get('role')
        content = msg.get('content')
        if role == 'user':
            avatar = '👤'
        else:
            avatar = '🤖'
            if isinstance(content, dict):
                msg_type = content.get('type')
                if msg_type == 'dataframe':
                    avatar = '📊'
                elif msg_type == 'map':
                    avatar = '🗺️'
                elif msg_type == 'code':
                    avatar = '💻'
                elif msg_type == 'log':
                    avatar = '📄'
        with st.chat_message(role, avatar=avatar):
            if isinstance(content, dict) and content.get('type') == 'dataframe':
                render_dataframe_with_viz(content['data'], key_prefix=f"df_{i}")
            elif isinstance(content, dict) and content.get('type') == 'map':
                map_df = content['data']
                layer = pdk.Layer(
                    'ScatterplotLayer',
                    data=map_df,
                    get_position='[lon, lat]',
                    get_radius=100,
                    get_fill_color='[255, 0, 0, 160]',
                    pickable=True
                )
                view_state = pdk.ViewState(
                    latitude=map_df['lat'].iloc[0],
                    longitude=map_df['lon'].iloc[0],
                    zoom=12,
                    pitch=0
                )
                tooltip = {"text": "📍 {name}"}
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))
            elif isinstance(content, dict) and content.get('type') == 'code':
                with st.expander(t('show_code'), expanded=False):
                    st.code(content['data'], language='sql')
                st.info(summarize_query(content['data'], selected_lang))
            elif isinstance(content, dict) and content.get('type') == 'log':
                with st.expander(t('sas_log_output')):
                    st.text_area("Log", content['data'], height=300)
            else:
                if isinstance(content, str):
                    st.markdown(
                        f"<div style='background-color:#f9f9f9;padding:8px;border-radius:6px;border:1px solid #e6e6e6;'>{content}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(content)

    if st.session_state.get('awaiting_ambiguity', False) and not st.session_state.get('awaiting_detail', False):
        current = st.session_state.get('current_ambiguity')
        queue = st.session_state.get('ambiguities_queue', [])
        if current and queue:
            label = f"{t('choose_correct_station')} '{current.get('term','?')}'"
            choice = st.selectbox(label=label,
                                options=current.get('options', []),
                                key=f"ambiguity_select_{len(queue)}")
            if st.button(t('confirm_selection'), key=f"confirm_ambiguity_button_{len(queue)}"):
                selected_station = choice
                combined_q = st.session_state.get('combined_question', st.session_state.get('last_user_input', ''))
                term = current.get('term') or ''
                try:
                    pattern = re.compile(re.escape(term), re.IGNORECASE)
                    combined_q = pattern.sub(selected_station, combined_q, count=1)
                except Exception:
                    combined_q = combined_q.replace(term, selected_station, 1)
                st.session_state['combined_question'] = combined_q

                # advance the queue
                queue.pop(0)
                if queue:
                    st.session_state['current_ambiguity'] = queue[0]
                    st.session_state['ambiguities_queue'] = queue
                    # keep awaiting_ambiguity=True and ask for next
                    st.session_state['messages'].append({'role': 'assistant', 'content': t('next_ambiguous')})
                else:
                    # all resolved
                    st.session_state['current_ambiguity'] = None
                    st.session_state['ambiguities_queue'] = []
                    st.session_state['awaiting_ambiguity'] = False

                    # proceed to validate & generate as before
                    with st.spinner(t('validating_question')):
                        reval = validate_question_llm(st.session_state['combined_question'])
                    if reval.get('error'):
                        st.session_state['messages'].append({'role': 'assistant', 'content': reval['error']})
                    else:
                        st.session_state['validated_question'] = reval.get('validated_question', st.session_state['combined_question'])
                        with st.spinner(t('generating_sql')):
                            query_result = generate_query_single_step(st.session_state['validated_question'])
                        query_text = (query_result.get('query') or '').strip()
                        st.session_state['generated_query'] = query_text
                        st.session_state['run_query_finished'] = False
                        # short memory of recent questions (max 20)
                        st.session_state['recent_questions'].append({'question': st.session_state['validated_question']})
                        if len(st.session_state['recent_questions']) > 20:
                            st.session_state['recent_questions'] = st.session_state['recent_questions'][-20:]
                        if query_text:
                            st.session_state['messages'].append({'role': 'assistant', 'content': {'type': 'code', 'data': query_text}})
                            _execute_current_query_now()
                            st.rerun()
                        else:
                            st.session_state['messages'].append({'role': 'assistant', 'content': t('enter_question_tip')})

                    # reset detail/ambiguity accumulators
                    st.session_state['pending_details'] = []
                    st.session_state['detail_values'] = {}
                    st.session_state['combined_question'] = ''
                    st.session_state['awaiting_detail'] = False

                st.rerun()

    # Chat input
    user_message = st.chat_input(t('enter_question'), key=chat_input_key)

    if user_message:
        st.session_state['messages'].append({'role': 'user', 'content': user_message})
        append_to_logfile(f"USER QUESTION: {user_message}")

        if st.session_state.get('awaiting_clarification', False):
            combined_q = st.session_state.get('combined_question', st.session_state.get('last_user_input', ''))
            # Append the user's reply if it isn't already part of the combined question
            if user_message.strip() not in combined_q:
                combined_q = (combined_q + ' ' + user_message.strip()).strip()
            st.session_state['combined_question'] = combined_q
            st.session_state['awaiting_clarification'] = False

            # Treat this combined question as the new user query
            question_for_validation = combined_q
            st.session_state['last_user_input'] = question_for_validation

            with st.spinner(t('validating_question')):
                validation_result = validate_question_llm(question_for_validation)

            # Handle validation outcomes similarly to the main flow
            if validation_result.get('error'):
                st.session_state['messages'].append({'role': 'assistant', 'content': validation_result['error']})
                st.rerun()
            else:
                clarification_msg = validation_result.get('clarification', '')
                missing_details = parse_missing_details(clarification_msg)
                ambiguities = _normalize_ambiguities(validation_result.get('ambiguities', []))

                # If there is yet another generic clarification, loop back by
                # setting awaiting_clarification and asking the follow‑up.
                if clarification_msg and not missing_details and not ambiguities:
                    st.session_state['combined_question'] = question_for_validation
                    st.session_state['awaiting_clarification'] = True
                    st.session_state['messages'].append({'role': 'assistant', 'content': clarification_msg})
                    st.rerun()

                # If structured details or ambiguities are required, set up
                # the appropriate state and prompts.
                if missing_details or ambiguities:
                    st.session_state['combined_question'] = question_for_validation
                    st.session_state['pending_details'] = missing_details
                    st.session_state['detail_values'] = {}

                    if ambiguities:
                        st.session_state['ambiguities_queue'] = ambiguities
                        st.session_state['current_ambiguity'] = ambiguities[0]
                        st.session_state['awaiting_ambiguity'] = True
                    else:
                        st.session_state['ambiguities_queue'] = []
                        st.session_state['current_ambiguity'] = None
                        st.session_state['awaiting_ambiguity'] = False

                    if missing_details:
                        first_detail = missing_details[0]
                        st.session_state['awaiting_detail'] = True
                        prompt_key = 'ask_train_type' if first_detail == 'type' else 'ask_date'
                        st.session_state['messages'].append({'role': 'assistant', 'content': t(prompt_key)})
                    else:
                        st.session_state['awaiting_detail'] = False
                        st.session_state['messages'].append({'role': 'assistant', 'content': t('ambiguous_station_msg')})
                    st.rerun()

                # Otherwise, we have a complete validated question – generate and execute the SQL
                st.session_state['validated_question'] = validation_result.get('validated_question', question_for_validation)
                with st.spinner(t('generating_sql')):
                    query_result = generate_query_single_step(st.session_state['validated_question'])
                query_text = (query_result.get('query') or '').strip()
                st.session_state['generated_query'] = query_text
                st.session_state['run_query_finished'] = False
                st.session_state['recent_questions'].append({'question': st.session_state['validated_question']})
                if len(st.session_state['recent_questions']) > 20:
                    st.session_state['recent_questions'] = st.session_state['recent_questions'][-20:]
                if query_text:
                    st.session_state['messages'].append({'role': 'assistant', 'content': {'type': 'code', 'data': query_text}})
                    _execute_current_query_now()
                    st.rerun()
                else:
                    st.session_state['messages'].append({'role': 'assistant', 'content': t('enter_question_tip')})

                # Reset temporary state flags after handling
                st.session_state['pending_details'] = []
                st.session_state['detail_values'] = {}
                st.session_state['combined_question'] = ''
                st.session_state['awaiting_detail'] = False
                st.session_state['awaiting_ambiguity'] = False
                st.rerun()

        if st.session_state.get('awaiting_location_question', False):
            enriched = enrich_question_with_locations(
                user_message,
                st.session_state.get('last_selected_locations', [])
            )
            st.session_state['awaiting_location_question'] = False

            # Incorporate recent conversation to make the question standalone
            question_for_validation = rewrite_with_history(enriched, selected_lang)
            st.session_state['last_user_input'] = question_for_validation
            with st.spinner(t('validating_question')):
                validation_result = validate_question_llm(question_for_validation)

            if validation_result.get('error'):
                st.session_state['messages'].append({'role': 'assistant', 'content': validation_result['error']})
                st.rerun()
            else:
                clarification_msg = validation_result.get('clarification', '')
                missing_details = parse_missing_details(clarification_msg)
                ambiguities = _normalize_ambiguities(validation_result.get('ambiguities', []))

                if clarification_msg and not missing_details and not ambiguities:
                    st.session_state['combined_question'] = question_for_validation
                    st.session_state['awaiting_clarification'] = True
                    st.session_state['messages'].append({'role': 'assistant', 'content': clarification_msg})
                    st.rerun()

                if missing_details or ambiguities:
                    st.session_state['combined_question'] = question_for_validation
                    st.session_state['pending_details'] = missing_details
                    st.session_state['detail_values'] = {}
                    if ambiguities:
                        st.session_state['ambiguities_queue'] = ambiguities
                        st.session_state['current_ambiguity'] = ambiguities[0]
                        st.session_state['awaiting_ambiguity'] = True
                    else:
                        st.session_state['ambiguities_queue'] = []
                        st.session_state['current_ambiguity'] = None
                        st.session_state['awaiting_ambiguity'] = False

                    if missing_details:
                        first_detail = missing_details[0]
                        st.session_state['awaiting_detail'] = True
                        prompt_key = 'ask_train_type' if first_detail == 'type' else 'ask_date'
                        st.session_state['messages'].append({'role': 'assistant', 'content': t(prompt_key)})
                    else:
                        st.session_state['awaiting_detail'] = False
                        st.session_state['messages'].append({'role': 'assistant', 'content': t('ambiguous_station_msg')})
                    st.rerun()
                else:
                    st.session_state['validated_question'] = validation_result.get('validated_question', question_for_validation)
                    with st.spinner(t('generating_sql')):
                        query_result = generate_query_single_step(st.session_state['validated_question'])
                    query_text = (query_result.get('query') or '').strip()
                    st.session_state['generated_query'] = query_text
                    st.session_state['run_query_finished'] = False
                    st.session_state['recent_questions'].append({'question': st.session_state['validated_question']})
                    if len(st.session_state['recent_questions']) > 20:
                        st.session_state['recent_questions'] = st.session_state['recent_questions'][-20:]
                    if query_text:
                        st.session_state['messages'].append({'role': 'assistant', 'content': {'type': 'code', 'data': query_text}})
                        _execute_current_query_now()
                        st.rerun()
                    else:
                        st.session_state['messages'].append({'role': 'assistant', 'content': t('enter_question_tip')})

                    st.session_state['pending_details'] = []
                    st.session_state['detail_values'] = {}
                    st.session_state['combined_question'] = ''
                    st.session_state['awaiting_detail'] = False
                    st.session_state['awaiting_ambiguity'] = False
                    st.rerun()

        if st.session_state.get('awaiting_detail', False):
            pending = st.session_state.get('pending_details', [])
            if pending:
                current_detail = pending[0]
                st.session_state['detail_values'][current_detail] = user_message.strip()
                combined_q = st.session_state.get('combined_question', st.session_state.get('last_user_input', ''))
                if user_message.strip() not in combined_q:
                    combined_q = (combined_q + ' ' + user_message.strip()).strip()
                st.session_state['combined_question'] = combined_q
                pending.pop(0)
                st.session_state['pending_details'] = pending
                if pending:
                    next_detail = pending[0]
                    prompt_key = 'ask_train_type' if next_detail == 'type' else 'ask_date'
                    st.session_state['messages'].append({'role': 'assistant', 'content': t(prompt_key)})
                    st.session_state['awaiting_detail'] = True
                else:
                    st.session_state['awaiting_detail'] = False
                    if st.session_state.get('awaiting_ambiguity', False):
                        st.session_state['messages'].append({
                            'role': 'assistant',
                            'content': t('ambiguous_station_msg')
                        })
                    else:
                        final_question = st.session_state.get('combined_question', '')
                        with st.spinner(t('validating_question')):
                            reval = validate_question_llm(final_question)
                        if reval.get('error'):
                            st.session_state['messages'].append({'role': 'assistant', 'content': reval['error']})
                        else:
                            st.session_state['validated_question'] = reval.get('validated_question', final_question)
                            with st.spinner(t('generating_sql')):
                                query_result = generate_query_single_step(st.session_state['validated_question'])
                            query_text = (query_result.get('query') or '').strip()
                            st.session_state['generated_query'] = query_text
                            st.session_state['run_query_finished'] = False
                            st.session_state['recent_questions'].append({'question': st.session_state['validated_question']})
                            if len(st.session_state['recent_questions']) > 20:
                                st.session_state['recent_questions'] = st.session_state['recent_questions'][-20:]
                            if query_text:
                                st.session_state['messages'].append({'role': 'assistant', 'content': {'type': 'code', 'data': query_text}})
                                _execute_current_query_now()
                                st.rerun()
                            else:
                                st.session_state['messages'].append({'role': 'assistant', 'content': t('enter_question_tip')})
                        st.session_state['pending_details'] = []
                        st.session_state['detail_values'] = {}
                        st.session_state['combined_question'] = ''
                        st.session_state['awaiting_detail'] = False
                        st.session_state['awaiting_ambiguity'] = False
                st.rerun()

        elif st.session_state.get('awaiting_ambiguity', False):
            raw = user_message.strip()
            current = st.session_state.get('current_ambiguity')
            queue = st.session_state.get('ambiguities_queue', [])

            if not current or not queue:
                st.session_state['awaiting_ambiguity'] = False
                st.rerun()

            options = current.get('options') or []
            typed_upper = raw.upper()

            selected_station = None
            if typed_upper in options:
                selected_station = typed_upper
            else:
                if raw.isdigit():
                    idx = int(raw) - 1
                    if 0 <= idx < len(options):
                        selected_station = options[idx]

            if not selected_station:
                st.session_state['messages'].append({
                    'role': 'assistant',
                    'content': (
                        t('choose_one_of')
                        + ", ".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
                    )
                })
                st.rerun()

            combined_q = st.session_state.get('combined_question', st.session_state.get('last_user_input', ''))
            term = (current.get('term') or '').strip()
            try:
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                combined_q = pattern.sub(selected_station, combined_q, count=1)
            except Exception:
                combined_q = combined_q.replace(term, selected_station, 1)

            st.session_state['combined_question'] = combined_q

            queue.pop(0)
            if queue:
                st.session_state['current_ambiguity'] = queue[0]
                st.session_state['ambiguities_queue'] = queue
                st.session_state['messages'].append({'role': 'assistant', 'content': t('next_ambiguous')})
                st.rerun()
            else:
                st.session_state['current_ambiguity'] = None
                st.session_state['ambiguities_queue'] = []
                st.session_state['awaiting_ambiguity'] = False

                with st.spinner(t('validating_question')):
                    reval = validate_question_llm(st.session_state['combined_question'])
                if reval.get('error'):
                    st.session_state['messages'].append({'role': 'assistant', 'content': reval['error']})
                    st.rerun()

                st.session_state['validated_question'] = reval.get('validated_question', st.session_state['combined_question'])

                with st.spinner(t('generating_sql')):
                    query_result = generate_query_single_step(st.session_state['validated_question'])
                query_text = (query_result.get('query') or '').strip()
                st.session_state['generated_query'] = query_text
                st.session_state['run_query_finished'] = False
                st.session_state['recent_questions'].append({'question': st.session_state['validated_question']})
                if len(st.session_state['recent_questions']) > 20:
                    st.session_state['recent_questions'] = st.session_state['recent_questions'][-20:]
                if query_text:
                    st.session_state['messages'].append({'role': 'assistant', 'content': {'type': 'code', 'data': query_text}})
                    _execute_current_query_now()
                    st.rerun()
                else:
                    st.session_state['messages'].append({'role': 'assistant', 'content': t('enter_question_tip')})

                st.session_state['pending_details'] = []
                st.session_state['detail_values'] = {}
                st.session_state['combined_question'] = ''
                st.session_state['awaiting_detail'] = False

                st.rerun()

        else:
            # Use conversation memory to rewrite follow-ups into standalone questions
            question_for_validation = rewrite_with_history(user_message, selected_lang)

            try:
                if _detect_follow_up(question_for_validation) and st.session_state.get('recent_questions'):
                    last_q_entry = st.session_state['recent_questions'][-1] if st.session_state['recent_questions'] else None
                    if last_q_entry and isinstance(last_q_entry, dict):
                        prev_q = last_q_entry.get('question', '')
                        if prev_q:
                            question_for_validation = _apply_follow_up_to_question(question_for_validation, prev_q)
            except Exception:
                pass
            st.session_state['last_user_input'] = question_for_validation
            with st.spinner(t('validating_question')):
                validation_result = validate_question_llm(question_for_validation)
            if validation_result.get('error'):
                st.session_state['messages'].append({'role': 'assistant', 'content': validation_result['error']})
                st.rerun()
            else:
                clarification_msg = validation_result.get('clarification', '')
                missing_details = parse_missing_details(clarification_msg)
                ambiguities = _normalize_ambiguities(validation_result.get('ambiguities', []))
                if clarification_msg and not missing_details and not ambiguities:
                    st.session_state['combined_question'] = question_for_validation
                    st.session_state['awaiting_clarification'] = True
                    st.session_state['messages'].append({'role': 'assistant', 'content': clarification_msg})
                    st.rerun()
                if missing_details or ambiguities:
                    st.session_state['combined_question'] = question_for_validation
                    st.session_state['pending_details'] = missing_details
                    st.session_state['detail_values'] = {}

                    if ambiguities:
                        st.session_state['ambiguities_queue'] = ambiguities
                        st.session_state['current_ambiguity'] = st.session_state['ambiguities_queue'][0]
                        st.session_state['awaiting_ambiguity'] = True
                    else:
                        st.session_state['ambiguities_queue'] = []
                        st.session_state['current_ambiguity'] = None
                        st.session_state['awaiting_ambiguity'] = False

                    if missing_details:
                        first_detail = missing_details[0]
                        st.session_state['awaiting_detail'] = True
                        prompt_key = 'ask_train_type' if first_detail == 'type' else 'ask_date'
                        st.session_state['messages'].append({'role': 'assistant', 'content': t(prompt_key)})
                    else:

                        st.session_state['awaiting_detail'] = False
                        st.session_state['messages'].append({'role': 'assistant', 'content': t('ambiguous_station_msg')})
                    st.rerun()
                else:

                    st.session_state['validated_question'] = validation_result.get('validated_question', question_for_validation)

                    with st.spinner(t('generating_sql')):
                        query_result = generate_query_single_step(st.session_state['validated_question'])
                    query_text = (query_result.get('query') or '').strip()
                    st.session_state['generated_query'] = query_text
                    st.session_state['run_query_finished'] = False
                    if query_text:
                        st.session_state['messages'].append({'role': 'assistant', 'content': {'type': 'code', 'data': query_text}})
                        _execute_current_query_now()
                        st.rerun()
                    else:
                        st.session_state['messages'].append({'role': 'assistant', 'content': t('enter_question_tip')})

                    st.session_state['pending_details'] = []
                    st.session_state['detail_values'] = {}
                    st.session_state['combined_question'] = ''
                    st.session_state['awaiting_detail'] = False
                    st.session_state['awaiting_ambiguity'] = False
                    st.rerun()

    with st.sidebar:
        # Quick controls
        if st.button('Clear chat'):
            st.session_state['messages'] = []
            st.session_state['recent_questions'] = []
            st.session_state['combined_question'] = ''
            st.session_state['awaiting_clarification'] = False
            st.session_state['awaiting_ambiguity'] = False
            st.session_state['awaiting_detail'] = False
            st.session_state['pending_details'] = []
            st.session_state['detail_values'] = {}
            st.rerun()

        with st.expander(t('query_history'), expanded=True):
            if not st.session_state['query_history']:
                st.write(t('no_history'))
            else:
                # show newest first
                for i, entry in enumerate(reversed(st.session_state['query_history'])):
                    index = len(st.session_state['query_history']) - i - 1
                    st.markdown(f"**{index + 1}. {entry.get('question','').strip()}**")

                    try:
                        df_hist = entry.get('data')
                        if df_hist is not None:
                            # small preview to keep sidebar compact
                            st.dataframe(df_hist.head(10), use_container_width=True, height=240)

                            try:
                                pdf_bytes = _dataframe_to_pdf_bytes(
                                    entry.get('question', '').strip() or 'Result',
                                    df_hist
                                )
                                st.download_button(
                                    label=t('download_pdf'),
                                    data=pdf_bytes,
                                    file_name=f"result_{index+1}.pdf",
                                    mime="application/pdf",
                                    key=f"dl_hist_pdf_{index}"
                                )
                            except Exception as pdf_err:
                                st.caption(f"(Could not generate PDF: {pdf_err})")

                    except Exception as e:
                        st.caption(f"Rendering failed {e}")

            if st.button(t('clear_history')):
                st.session_state['query_history'] = []
                st.rerun()


if __name__ == '__main__':
    main()
