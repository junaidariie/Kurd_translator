import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

BASE = "facebook/nllb-200-distilled-600M"
LORA = "junaid17/nllb-kurdish-lora"

tokenizer = AutoTokenizer.from_pretrained(BASE)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None

def load_model():
    global _model
    if _model is None:
        try:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE)
            _model = PeftModel.from_pretrained(base_model, LORA).eval()
            print("Model loaded succesfully...")
        except Exception as e:
            print(f"Error while loading the model : {str(e)}")
    return _model.to(device)

#model = load_model()

def translate(src_lang, tgt_lang, model, text):
    try:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)

        output_tokens = model.generate(
            **encoded,
            forced_bos_token_id=forced_bos,
            max_length=256,
            num_beams=4
        )

        return tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Could't translate due to unexpected error : {str(e)}")



#text = "hello, my name is junaid"
#print(translate(src_lang='eng_Latn', tgt_lang='ckb_Arab', model=model, text=text))
