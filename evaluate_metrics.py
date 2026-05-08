import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm
import re

# -------------------------------
# CONFIG
# -----------------------------y--
MODEL_CHECKPOINT = "ai4bharat/IndicBERTv2-MLM-Sam-TLM"
SAVED_MODEL_PATH = "cyberbully_model"
TEST_DATA_PATH = "2k_Dataset.csv" # Your test dataset file

# -------------------------------
# PREPROCESSING
# -------------------------------
def minimal_preprocessing(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------------
# LOAD MODEL
# -------------------------------
print("Loading Model and Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(SAVED_MODEL_PATH)

base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=2,
    id2label={0: "Non-Bullying", 1: "Bullying"},
    label2id={"Non-Bullying": 0, "Bullying": 1},
)

model = PeftModel.from_pretrained(base_model, SAVED_MODEL_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded successfully on {device}!")

# -------------------------------
# EVALUATION LOGIC
# -------------------------------
def predict(text):
    clean_text = minimal_preprocessing(text)
    inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    probs = F.softmax(logits, dim=-1)
    pred_id = torch.argmax(probs, dim=-1).item()
    return pred_id

# -------------------------------
# RUN EVALUATION
# -------------------------------
print("Loading Test Data...")
df = pd.read_csv(TEST_DATA_PATH)

# Ensure columns exist
if not all(col in df.columns for col in ['text', 'label']):
    raise ValueError("CSV must contain 'text' and 'label' columns")

y_true = df['label'].tolist()
y_pred = []

print("Running predictions on test set...")
# tqdm gives a nice progress bar
for text in tqdm(df['text'].tolist()):
    y_pred.append(predict(text))

# -------------------------------
# PRINT OVERALL METRICS (For Table 3)
# -------------------------------
print("\n" + "="*50)
print("OVERALL METRICS (TABLE 3)")
print("="*50)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Non-Bullying", "Bullying"]))

# -------------------------------
# PRINT LANGUAGE-WISE METRICS (For Table 4)
# -------------------------------
if 'language' in df.columns:
    print("\n" + "="*50)
    print("LANGUAGE-WISE F1-SCORES (TABLE 4)")
    print("="*50)
    
    languages = df['language'].unique()
    for lang in languages:
        # Filter data for each language
        lang_df = df[df['language'] == lang]
        lang_true = lang_df['label'].tolist()
        
        # Get predictions just for this language
        lang_pred = [y_pred[i] for i in lang_df.index]
        
        lang_f1 = f1_score(lang_true, lang_pred)
        print(f"{lang:15} : F1-Score = {lang_f1:.4f}")
else:
    print("\nTip: Add a 'language' column to your CSV to get Table 4 metrics!")