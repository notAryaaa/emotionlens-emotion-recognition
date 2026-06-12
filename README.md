# 🎭 EmotionLens — Text Emotion Recognition

A web application that detects emotions in text using two models side-by-side: a classical ML baseline and a fine-tuned BERT transformer.

> **NLP Course Project — Istanbul Medipol University, Spring 2026**
> Arya Ghazizadeh (64210017) · Mohammad Shafizadeh (64210053)

---

## 🎯 What It Does

You type any sentence. EmotionLens instantly classifies it into one of **6 emotions**:

| Emotion | Emoji |
|---------|-------|
| Sadness | 😢 |
| Joy | 😄 |
| Love | ❤️ |
| Anger | 😠 |
| Fear | 😨 |
| Surprise | 😮 |

Both models run simultaneously so you can compare their predictions and confidence scores in real time.

---

## 📊 Results

| Metric | Baseline (TF-IDF + LogReg) | BERT (fine-tuned) |
|--------|---------------------------|-------------------|
| Accuracy | 86.9% | **92.7%** |
| Macro F1 | 0.811 | **0.878** |
| Macro Recall | 0.795 | **0.875** |
| Weighted F1 | 0.867 | **0.927** |

BERT outperforms the baseline across every metric. The biggest per-class gain is **Love (+12.2% F1)** — where contextual understanding matters most.

---

## 🏗️ Architecture

```
User Input (text)
       │
       ├──► Baseline Model          ──► TF-IDF (10K features, bigrams)
       │    baseline_model.pkl           + Logistic Regression (C=10)
       │
       └──► BERT Model              ──► bert-base-uncased (110M params)
            bert-emotion-saved/          fine-tuned · 4 epochs · A100 GPU
```

Both predictions are served through a Flask API and displayed side-by-side with animated confidence bars.

---

## 📁 Project Structure

```
emotionlens-emotion-recognition/
├── app.py                          # Flask web app + inference logic
├── templates/
│   └── index.html                  # Frontend UI
├── Emotion_Recognition_NLP_Full.ipynb   # Training notebook (Google Colab)
├── requirements.txt
└── README.md
```

> **Model files** are not included in this repo due to size limits.
> Generate them by running the Colab notebook (see Setup below).

---

## ⚙️ Setup

### Prerequisites
- Python 3.9+
- A Google account (to run the Colab notebook)

### Step 1 — Train the models (Google Colab)

1. Open `Emotion_Recognition_NLP_Full.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU**: Runtime → Change runtime type → T4 GPU
3. Run all cells: **Ctrl+F9**
4. The last cell downloads `results.zip` — extract it
5. You will get:
   - `baseline_model.pkl`
   - `bert-emotion-saved/` folder

### Step 2 — Set up the web app

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/emotionlens-emotion-recognition.git
cd emotionlens-emotion-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 3 — Add model files

Place the files from Step 1 into the project folder:

```
emotionlens-emotion-recognition/
├── baseline_model.pkl              ← from Colab
├── bert-emotion-saved/             ← from Colab
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.txt
```

### Step 4 — Run

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🧪 Dataset

**dair-ai/emotion** — 20,000 English tweets annotated with 6 emotion labels.

| Split | Samples |
|-------|---------|
| Train | 16,000 |
| Validation | 2,000 |
| Test | 2,000 |

Loaded automatically from HuggingFace during notebook training:
```python
from datasets import load_dataset
dataset = load_dataset('dair-ai/emotion')
```

---

## 🔬 Models

### Baseline — TF-IDF + Logistic Regression
- TF-IDF: 10,000 features, unigram + bigram, sublinear TF scaling
- Logistic Regression: multinomial, best C=10 (via GridSearchCV)
- Training time: ~30 seconds on CPU

### BERT — Fine-tuned bert-base-uncased
- Base model: `bert-base-uncased` (110M parameters)
- Max sequence length: 128 tokens
- Training: 4 epochs, batch size 32, lr 2e-5, AdamW, weight decay 0.01
- Hardware: NVIDIA A100 GPU, FP16 mixed precision
- Training time: ~2 minutes

---

## 📦 Dependencies

```
flask
torch
transformers
scikit-learn
joblib
numpy
datasets
```

Install all with: `pip install -r requirements.txt`
