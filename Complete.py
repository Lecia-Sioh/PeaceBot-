# !pip install streamlit transformers sentencepiece langdetect openpyxl rapidfuzz

import streamlit as st
import datetime
import pandas as pd
import re
import logging
from transformers import pipeline, MarianMTModel, MarianTokenizer
from langdetect import detect
from rapidfuzz import fuzz

# === MUST BE FIRST STREAMLIT COMMAND ===
st.set_page_config(page_title="PeaceBot", page_icon="🕊️")

# === CONFIGURATION ===
ADMIN_PASSWORD = "admin123"
EXCEL_FILE = "reports.xlsx"

BLOCKED_WORDS = [
    "idiot", "stupid", "dumb", "moron", "retard", "loser", "fool",
    "bastard", "bitch", "slut", "whore", "asshole", "dick", "pussy",
    "fuck", "shit", "crap", "damn", "hell", "cunt", "prick", "twat",
    "nigger", "chink", "spic", "fag", "dyke", "kike", "tranny",
    "beaner", "coon", "terrorist", "raghead", "shemale", "pedo", "rapist",
    "kill", "die", "hang", "lynch", "burn", "exterminate", "bomb", "shoot",
    "gas", "molest", "abuse", "slave", "hitler", "nazi", "cock", "dickhead", 
    "motherfuckers", "bitches", "smelly", "bozo", "skank", "NSFW", "FML", 
    "TMI", "IDC", "WTF", "TL;DR", "hoe", "douche", "dumbass", "shithead", "jerk"
]

WHITELIST = {
    "good", "best", "hello", "happy", "friend", "nice", "love", "fine", "thanks",
    "you", "me", "a", "okay", "peace", "cool", "welcome", "great", "hi", "well", "the", "think",
    "it", "an"
}

FUZZY_MATCH_THRESHOLD = 85

# === Logging Setup ===
logging.basicConfig(level=logging.WARNING)

# === Session Initialization ===
if 'tone_analyzer' not in st.session_state:
    st.session_state.tone_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

if 'toxicity_analyzer' not in st.session_state:
    st.session_state.toxicity_analyzer = pipeline("text-classification", model="unitary/toxic-bert")

if 'report_data' not in st.session_state:
    try:
        df_existing = pd.read_excel(EXCEL_FILE)
        st.session_state.report_data = df_existing.to_dict("records")
    except FileNotFoundError:
        st.session_state.report_data = []

if 'admin_logged_in' not in st.session_state:
    st.session_state.admin_logged_in = False

# === Language Map ===
lang_code_map = {
    "Chinese": "zh",
    "Hindi": "hi"
}

def normalize_lang_code(code):
    mapping = {
        "zh-cn": "zh", "zh": "zh", "en": "en", "hi": "hi",
        "ms": "hi", "mt": "hi", "sw": "hi"
    }
    return mapping.get(code.lower(), code.lower())

def clean_subtitle_tags(text):
    return re.sub(r"\{.*?\}", "", text).strip()

@st.cache_resource
def get_translation_model(src, tgt):
    src = normalize_lang_code(src)
    tgt = normalize_lang_code(tgt)
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate_text(text, src, tgt):
    try:
        src = normalize_lang_code(src)
        tgt = normalize_lang_code(tgt)
        if src == tgt:
            return text
        cleaned_text = clean_subtitle_tags(text)
        tokenizer, model = get_translation_model(src, tgt)
        batch = tokenizer.prepare_seq2seq_batch([cleaned_text], return_tensors="pt", truncation=True)
        gen = model.generate(**batch)
        return tokenizer.decode(gen[0], skip_special_tokens=True)
    except Exception as e:
        logging.warning(f"Translation failed: {e}")
        return "⚠️ Could not translate this message."

def is_similar_to_blocked(word, blocklist):
    word = word.lower()
    if len(word) <= 1 or word in WHITELIST:
        return False
    threshold = 90 if len(word) <= 4 else 85
    for bad_word in blocklist:
        if fuzz.partial_ratio(word, bad_word) >= threshold:
            return True
    return False

def censor_text(text, blocklist):
    words = text.split()
    censored = [("*" * len(w)) if is_similar_to_blocked(w, blocklist) else w for w in words]
    return " ".join(censored)

def analyze_tone_and_conflict(text):
    tone_result = st.session_state.tone_analyzer(text)[0]
    tone_label = tone_result["label"]
    tone_score = round(tone_result["score"] * 100, 2)

    conflict_result = st.session_state.toxicity_analyzer(text)[0]
    raw_conflict_label = conflict_result["label"].upper()
    conflict_score = round(conflict_result["score"] * 100, 2)

    # Fix: Change displayed label based on threshold
    conflict_label = "TOXIC" if conflict_score >= 50 else "NOT_TOXIC"

    final_score = max(tone_score, conflict_score)

    if conflict_score >= 80:
        level = "high"
        icon = "🔥"
        prefix = "High Conflict Risk"
    elif conflict_score >= 50:
        level = "moderate"
        icon = "⚠️"
        prefix = "Moderate Conflict Risk"
    else:
        level = "low"
        icon = "✅"
        prefix = "No Conflict Risk"

    combined_msg = f"**🧠 Tone:** {tone_label} | {icon} {prefix} ({conflict_label}) – Confidence: {final_score}%"

    return combined_msg, level

# === PeaceBot UI ===
st.title("🕊️ PeaceBot - Communication Assistant")
user_input = st.text_area("✍️ Enter your message below:", height=150)
target_lang_label = st.selectbox("🌍 Translate suggestion to:", ["None", "Chinese", "Hindi"])

if st.button("Analyze & Enhance"):
    if user_input:
        detected_lang = normalize_lang_code(detect(user_input))
        st.markdown(f"**🌐 Detected Language:** {detected_lang.upper()}")

        combined_msg, conflict_level = analyze_tone_and_conflict(user_input)

        if conflict_level == "high":
            st.error(combined_msg)
        elif conflict_level == "moderate":
            st.warning(combined_msg)
        else:
            st.success(combined_msg)

        censored = censor_text(user_input, BLOCKED_WORDS)
        flagged_words = [w for w in user_input.split() if is_similar_to_blocked(w, BLOCKED_WORDS)]

        st.write("**🔭️ Censored Message:**")
        if flagged_words:
            st.error("⚠️ Message contains offensive or sensitive words.")
            st.write(censored)
        else:
            st.success("✅ This message is clean.")
            st.write(user_input)

        if target_lang_label != "None":
            target_lang = lang_code_map[target_lang_label]
            if detected_lang != target_lang:
                translated = translate_text(censored, detected_lang, target_lang)
                st.write(f"**🌐 Translated to {target_lang_label}:**")
                st.success(translated)
            else:
                st.info("🌐 Input and target language are the same. Skipping translation.")

        st.markdown("### 🛡️ Concerned about a message? Submit an anonymous report below.")

# === Anonymous Reporting ===
st.markdown("---")
st.header("🚩 Anonymous Reporting System")
with st.form("report_form"):
    reason = st.selectbox("Reason", ["Harassment", "Hate Speech", "Exclusion", "Inappropriate Content", "Other"])
    reported = st.text_input("Who/What are you reporting?")
    description = st.text_area("Details (optional)", max_chars=500)
    submit = st.form_submit_button("Submit Report")
    if submit:
        if reported.strip():
            new_report = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "reason": reason,
                "reported": reported.strip(),
                "description": description.strip()
            }
            st.session_state.report_data.append(new_report)
            pd.DataFrame(st.session_state.report_data).to_excel(EXCEL_FILE, index=False)
            st.success("✅ Report submitted successfully.")
        else:
            st.warning("Please enter a valid report subject.")

# === Admin Panel ===
st.markdown("---")
if not st.session_state.admin_logged_in:
    with st.expander("🔐 Admin Login"):
        admin_pass = st.text_input("Enter admin password:", type="password")
        if st.button("Login"):
            if admin_pass == ADMIN_PASSWORD:
                st.session_state.admin_logged_in = True
                st.success("Access granted.")
            else:
                st.error("Incorrect password.")
else:
    st.subheader("🗂 Admin Report View")
    if st.session_state.report_data:
        st.dataframe(pd.DataFrame(st.session_state.report_data))
    else:
        st.info("No reports submitted yet.")
