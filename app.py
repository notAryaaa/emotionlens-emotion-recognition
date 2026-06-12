"""
Emotion Recognition Demo — Flask Web App
Run: python app.py
Then open http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import joblib
import torch
import numpy as np
import re
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# ── Config ──
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOTION_EMOJIS = {"sadness": "😢", "joy": "😄", "love": "❤️", "anger": "😠", "fear": "😨", "surprise": "😮"}
EMOTION_COLORS = {"sadness": "#5B8DEE", "joy": "#FFD93D", "love": "#FF6B6B", "anger": "#E74C3C", "fear": "#9B59B6", "surprise": "#1ABC9C"}

BERT_MODEL_DIR = "bert-emotion-saved"
BASELINE_MODEL_PATH = "baseline_model.pkl"

# ── Load Models ──
print("Loading models...")

# Baseline
baseline_model = None
if os.path.exists(BASELINE_MODEL_PATH):
    baseline_model = joblib.load(BASELINE_MODEL_PATH)
    print("✓ Baseline model loaded")
else:
    print(f"✗ Baseline model not found at {BASELINE_MODEL_PATH}")

# BERT
bert_model = None
bert_tokenizer = None
if os.path.exists(BERT_MODEL_DIR):
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
    bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
    bert_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    print(f"✓ BERT model loaded (device: {device})")
else:
    print(f"✗ BERT model not found at {BERT_MODEL_DIR}")

print("Ready!\n")


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict_baseline(text):
    if baseline_model is None:
        return None
    cleaned = clean_text(text)
    probs = baseline_model.predict_proba([cleaned])[0]
    pred_idx = int(np.argmax(probs))
    return {
        "label": EMOTION_LABELS[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probs": {EMOTION_LABELS[i]: round(float(probs[i]), 4) for i in range(len(EMOTION_LABELS))}
    }


def predict_bert(text):
    if bert_model is None or bert_tokenizer is None:
        return None
    device = next(bert_model.parameters()).device
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return {
        "label": EMOTION_LABELS[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probs": {EMOTION_LABELS[i]: round(float(probs[i]), 4) for i in range(len(EMOTION_LABELS))}
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = {
        "text": text,
        "baseline": predict_baseline(text),
        "bert": predict_bert(text),
        "emojis": EMOTION_EMOJIS,
        "colors": EMOTION_COLORS,
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
