import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from peft import PeftModel

st.set_page_config(page_title="Kurdish Translator", layout="centered")
st.title("Kurdish ↔ English Translator (NLLB + LoRA)")

# Model paths
BASE_MODEL = "facebook/nllb-200-distilled-600M"
ADAPTER_PATH = "./nllb_kurdish_lora_adapter"


@st.cache_resource
def load_translation_pipeline(direction):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Load base model
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    if direction == "English → Kurdish":
        src = "eng_Latn"
        tgt = "ckb_Arab"
    else:  # Kurdish → English
        src = "ckb_Arab"
        tgt = "eng_Latn"

    # Build translation pipeline
    translator = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang=src,
        tgt_lang=tgt,
        max_length=256
    )

    return translator


# UI: Translation direction
direction = st.selectbox(
    "Select Translation Direction",
    ["English → Kurdish", "Kurdish → English"]
)

# Text input box
text = st.text_area("Enter text below:", height=180)

# Translate Button
if st.button("Translate"):
    if text.strip():
        translator = load_translation_pipeline(direction)
        output = translator(text)[0]["translation_text"]
        st.subheader("Translation:")
        st.success(output)
    else:
        st.warning("Please enter some text to translate.")
