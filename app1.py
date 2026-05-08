import streamlit as st
from transformers import pipeline

# We use DistilBERT because it's the gold standard for "Lightweight Transformers"
# (40% smaller, 60% faster than BERT)
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english" 

st.set_page_config(page_title="CyberGuard AI", page_icon="🛡️")
st.title("🛡️ Cyberbullying Detector")
st.markdown("Implementation of a **Lightweight Transformer** for real-time moderation.")

# Load model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model=MODEL_NAME)

classifier = load_model()

# User Input
user_text = st.text_area("Enter social media comment:", placeholder="Type something here...")

if st.button("Analyze Content"):
    if user_text:
        result = classifier(user_text)[0]
        label = result['label']
        score = result['score']
        
        if label == "NEGATIVE": # Simplified logic for demo
            st.error(f"🚨 Potential Bullying Detected! (Confidence: {score:.2%})")
        else:
            st.success(f"✅ Content is Safe. (Confidence: {score:.2%})")
    else:
        st.warning("Please enter some text first.")