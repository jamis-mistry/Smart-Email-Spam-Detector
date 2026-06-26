# 🛡️ Smart Email Spam Detector

A real-time email spam detection web application built with **Streamlit** and **scikit-learn**.

## 🚀 Features

- **Real-time spam detection** — paste any email and get instant results
- **3 ML Models** — Naive Bayes, Logistic Regression, Linear SVM
- **Confidence scoring** with an interactive gauge chart
- **Feature analysis** — URL detection, spam keywords, urgency language, uppercase ratio
- **Model performance dashboard** — accuracy, precision, recall, F1, confusion matrix
- **Dataset analysis** — class distribution, word count distribution, sample data
- **How It Works** tab — full explanation of the NLP pipeline

## 📁 Project Structure

```
smart_spam_detector/
│
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## 🛠️ Setup & Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/smart-spam-detector.git
cd smart-spam-detector
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## ☁️ Deploy on Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** → `app.py`
5. Click **Deploy!**

## 🧠 ML Pipeline

```
Email Text
    ↓
Text Preprocessing (lowercase, URL tokenization, punctuation removal)
    ↓
TF-IDF Vectorization (1-gram + 2-gram, max 5000 features)
    ↓
ML Classifier (Naive Bayes / Logistic Regression / Linear SVM)
    ↓
Prediction + Confidence Score
    ↓
Rule-based Feature Analysis
    ↓
Final Result (SPAM / HAM)
```

## 📊 Models Used

| Model | Type | Probability | Speed |
|---|---|---|---|
| Naive Bayes | Probabilistic | ✅ Yes | ⚡ Very Fast |
| Logistic Regression | Linear | ✅ Yes | ⚡ Fast |
| Linear SVM | Margin-based | ❌ No | ⚡ Fast |

## 📌 Technologies

- **Python 3.10+**
- **Streamlit** — Web framework
- **scikit-learn** — ML models & TF-IDF
- **Plotly** — Interactive charts
- **Pandas / NumPy** — Data processing

## 👤 Author

Jamis Mistry — [GitHub](https://github.com/jamis-mistry)

Naivedh Jariwala — [GitHub](https://github.com/naivedhjariwala)

---
*Built as an academic project demonstrating NLP + ML for email spam classification.*
