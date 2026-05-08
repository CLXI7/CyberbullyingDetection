import streamlit as st
import torch
import re
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# -------------------------------
# CONFIG
# -------------------------------
MODEL_CHECKPOINT = "ai4bharat/IndicBERTv2-MLM-Sam-TLM"
SAVED_MODEL_PATH = "cyberbully_model"

st.set_page_config(page_title="Cyberbullying Detection", layout="centered")

# -------------------------------
# PREPROCESSING
# -------------------------------
def minimal_preprocessing(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------------
# LOAD MODEL (cached)
# -------------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(SAVED_MODEL_PATH)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=2,
        id2label={0: "Non-Bullying", 1: "Bullying"},
        label2id={"Non-Bullying": 0, "Bullying": 1},
        weights_only=False
    )

    model = PeftModel.from_pretrained(base_model, SAVED_MODEL_PATH)
    model.eval()

    return model, tokenizer

model, tokenizer = load_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------------------
# PREDICTION + EXPLAINABILITY
# -------------------------------
def predict_and_explain(text):
    clean_text = minimal_preprocessing(text)

    inputs = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        logits = outputs.logits
        attentions = outputs.attentions

    probs = F.softmax(logits, dim=-1)
    pred_id = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred_id].item()
    label = model.config.id2label[pred_id]

    # Explainability
    last_attention = attentions[-1]
    avg_attention = last_attention.mean(dim=1)
    cls_attention = avg_attention[0, 0, :].cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    token_scores = [
        (tok, score)
        for tok, score in zip(tokens, cls_attention)
        if tok not in tokenizer.all_special_tokens
    ]

    token_scores.sort(key=lambda x: x[1], reverse=True)

    return label, confidence, token_scores[:5]

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("🛡️ Cyberbullying Detection (IndicBERT + LoRA)")
st.markdown("Supports **Malayalam, Manglish & English**")

text_input = st.text_area(
    "Enter a comment:",
    placeholder="ninakku vattano, eda bhranthan."
)

if st.button("Analyze"):
    if text_input.strip() == "":
        st.warning("Please enter some text")
    else:
        with st.spinner("Analyzing..."):
            label, confidence, tokens = predict_and_explain(text_input)

        if label == "Bullying":
            st.error(f"🚨 **Bullying Detected**")
        else:
            st.success(f"✅ **Non-Bullying**")

        st.metric("Confidence", f"{confidence:.2%}")

        st.subheader("🔍 Influential Words")
        for tok, score in tokens:
            st.write(f"**{tok}** → {score:.4f}")
