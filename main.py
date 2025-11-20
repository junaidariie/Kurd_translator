import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from peft import PeftModel

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Kurdish Translator", layout="wide")

# Initialize session state
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "src_lang" not in st.session_state:
    st.session_state.src_lang = "English"
if "tgt_lang" not in st.session_state:
    st.session_state.tgt_lang = "Kurdish"
if "output_text" not in st.session_state:
    st.session_state.output_text = ""

# Theme colors
THEME = st.session_state.theme
bg = "#111111" if THEME == "dark" else "#ffffff"
fg = "#ffffff" if THEME == "dark" else "#000000"
card_bg = "rgba(255,255,255,0.07)" if THEME == "dark" else "rgba(0,0,0,0.05)"

st.markdown(
    f"""<style>
body {{
    background-color: {bg};
    color: {fg};
}}
.dots-loader {{
    display: flex;
    justify-content: center;
    margin-top: 10px;
}}
.dots-loader div {{
    width: 12px;
    height: 12px;
    margin: 4px;
    background-color: #4A90E2;
    border-radius: 50%;
    animation: bounce 0.6s infinite alternate;
}}
.dots-loader div:nth-child(2) {{ animation-delay: 0.2s; }}
.dots-loader div:nth-child(3) {{ animation-delay: 0.4s; }}
@keyframes bounce {{
    from {{ transform: translateY(0); }}
    to {{ transform: translateY(-12px); }}
}}
.output-box {{
    padding: 20px;
    border-radius: 12px;
    background: {card_bg};
    font-size: 18px;
    min-height: 120px;
    word-wrap: break-word;
}}
</style>""",
    unsafe_allow_html=True
)

# -----------------------------
# LOAD MODEL
# -----------------------------
BASE_MODEL = "facebook/nllb-200-distilled-600M"
LORA_REPO = "junaid17/nllb-kurdish-lora"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base, LORA_REPO)
    return model, tokenizer

model, tokenizer = load_model()

# -----------------------------
# TRANSLATION FUNCTION
# -----------------------------
def translate(text, src, tgt):
    translator = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang=src,
        tgt_lang=tgt,
        max_length=256,
    )
    return translator(text)[0]["translation_text"]

# -----------------------------
# CALLBACK FUNCTIONS
# -----------------------------
def swap_languages():
    st.session_state.src_lang, st.session_state.tgt_lang = (
        st.session_state.tgt_lang,
        st.session_state.src_lang
    )

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

def copy_to_clipboard():
    if st.session_state.output_text:
        st.toast("✓ Text copied to clipboard!", icon="✅")

# -----------------------------
# UI
# -----------------------------
st.title("Kurdish ↔ English Translator (NLLB + LoRA)")

col1, colSwap, colTheme, col2 = st.columns([1, 0.6, 0.8, 1])

with col1:
    src_lang = st.selectbox(
        "From:",
        ["English", "Kurdish"],
        key="src_lang"
    )

with colSwap:
    st.button("⇆ Swap", use_container_width=True, on_click=swap_languages)

with colTheme:
    theme_icon = "🌙" if THEME == "light" else "☀️"
    st.button(theme_icon, use_container_width=True, on_click=toggle_theme)

with col2:
    tgt_lang = st.selectbox(
        "To:",
        ["Kurdish", "English"],
        key="tgt_lang"
    )

lang_codes = {
    "English": "eng_Latn",
    "Kurdish": "ckb_Arab"
}

src_code = lang_codes[st.session_state.src_lang]
tgt_code = lang_codes[st.session_state.tgt_lang]

text = st.text_area("Enter text:", height=180)

# -----------------------------
# TRANSLATE BUTTON
# -----------------------------
if st.button("Translate", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Translating..."):
            try:
                output = translate(text, src_code, tgt_code)
                st.session_state.output_text = output
            except Exception as e:
                st.error(f"Translation error: {str(e)}")
                st.session_state.output_text = ""

# Display output if available
if st.session_state.output_text:
    st.subheader("Output")
    st.markdown(
        f"<div class='output-box'>{st.session_state.output_text}</div>",
        unsafe_allow_html=True
    )
    
    # Copy button with pyperclip alternative
    col_copy, col_empty = st.columns([1, 3])
    with col_copy:
        # Display the output in a code block that users can easily select and copy
        st.code(st.session_state.output_text, language=None)
        st.caption("👆 Select and copy the text above (Ctrl+C / Cmd+C)")
