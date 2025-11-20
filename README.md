# Kurdish ↔ English Neural Machine Translation (NLLB + LoRA Fine-Tuning)

## 📌 Project Overview
This project implements a **Kurdish ↔ English neural machine translation system** using Meta’s **NLLB-200 (No Language Left Behind)** model.  
The base model `facebook/nllb-200-distilled-600M` was fine-tuned using **LoRA (Low-Rank Adaptation)** on a curated Kurdish–English dataset.

The system supports:

- **English → Kurdish (Sorani – ckb_Arab)**  
- **Kurdish → English**

This project includes:

- A fine-tuned LoRA adapter  
- A Streamlit translator UI  
- A HuggingFace Space deployment  

---

## 🚀 Live Demo
https://huggingface.co/spaces/junaid17/translator

---

## 🧠 Model Details

### Base Model
- `facebook/nllb-200-distilled-600M`
- Supports 200+ languages
- 600M parameters

### Fine-Tuned Adapter
- LoRA fine-tuning on 20K Kurdish–English samples
- Parameter-efficient training
- Hosted model: https://huggingface.co/junaid17/nllb-kurdish-lora

---

## 🛠 Technologies Used
- PyTorch  
- HuggingFace Transformers  
- PEFT (LoRA)  
- Streamlit  
- HuggingFace Spaces  

---

## 📦 How to Use the Model in Python
*(Section intentionally left empty per request.)*

---

## 🌐 Streamlit Web App
Features:

- Two-way translation  
- Language swap button  
- Dark/Light mode toggle  
- Animated loader  
- Copy-to-clipboard  
- Caching to avoid reloading the model  
- Modern clean UI  

---

## 📁 Project Structure
```
project/
│── main.py
│── requirements.txt
│── README.md
```

---

## 🎯 Key Features
- Bidirectional translation  
- Lightweight LoRA adapter  
- Works on CPU  
- Real-world deployment  
- Uses NLLB language codes:
  - `eng_Latn`  
  - `ckb_Arab`

---

## 📣 Why This Project Stands Out
- Kurdish is a low-resource language  
- Fine-tuning NLLB is advanced  
- End-to-end ML engineering and deployment  
- Production-level UI and performance  

---

## 📬 Author
**Junaid** — AI/ML Engineer & Deep Learning Practitioner.
