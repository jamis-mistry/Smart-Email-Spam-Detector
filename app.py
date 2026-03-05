import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import pickle
import os
import time
from datetime import datetime

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)
from sklearn.pipeline import Pipeline

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Smart Email Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #0f1629 50%, #0d0d1a 100%);
    color: #e0e6ff;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a18 0%, #111827 100%);
    border-right: 1px solid #1e2a4a;
}
[data-testid="stSidebar"] * { color: #c4cfea !important; }

/* Headers */
h1 { font-family: 'Space Mono', monospace !important; color: #7c9dff !important; }
h2, h3 { color: #a5b8ff !important; }

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #131b35, #1a2347);
    border: 1px solid #2a3a6a;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    margin: 8px 0;
    box-shadow: 0 4px 24px rgba(100,130,255,0.08);
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-value { font-size: 2.2rem; font-weight: 700; color: #7c9dff; font-family: 'Space Mono', monospace; }
.metric-label { font-size: 0.82rem; color: #8899cc; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 6px; }

/* Result boxes */
.spam-box {
    background: linear-gradient(135deg, #2d0a0a, #3d1010);
    border: 2px solid #e05555;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    animation: pulse-red 2s infinite;
}
.ham-box {
    background: linear-gradient(135deg, #0a2d0a, #103d10);
    border: 2px solid #44cc66;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    animation: pulse-green 2s infinite;
}
@keyframes pulse-red  { 0%,100%{box-shadow:0 0 0 0 rgba(224,85,85,.4)}  50%{box-shadow:0 0 0 12px rgba(224,85,85,0)} }
@keyframes pulse-green{ 0%,100%{box-shadow:0 0 0 0 rgba(68,204,102,.4)} 50%{box-shadow:0 0 0 12px rgba(68,204,102,0)} }

.result-emoji { font-size: 4rem; }
.result-title { font-size: 1.8rem; font-weight: 700; margin: 8px 0; font-family: 'Space Mono', monospace; }
.result-conf  { font-size: 1rem; opacity: 0.8; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3a5bff, #6a3aff) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 16px rgba(58,91,255,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(58,91,255,0.5) !important;
}

/* Text area */
.stTextArea textarea {
    background: #0e152e !important;
    border: 1px solid #2a3a6a !important;
    color: #e0e6ff !important;
    border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #0a0e1f; border-radius: 10px; gap: 4px; }
.stTabs [data-baseweb="tab"] { color: #8899cc; border-radius: 8px; }
.stTabs [aria-selected="true"] { background: #1a2347 !important; color: #7c9dff !important; }

/* Divider */
hr { border-color: #1e2a4a !important; }

/* Info boxes */
.stInfo { background: #0e1a3a !important; border-left-color: #3a5bff !important; }

/* Tag pill */
.spam-tag { background:#3d1010; color:#e05555; border:1px solid #e05555; border-radius:20px; padding:3px 12px; font-size:0.78rem; display:inline-block; }
.ham-tag  { background:#103d10; color:#44cc66; border:1px solid #44cc66; border-radius:20px; padding:3px 12px; font-size:0.78rem; display:inline-block; }

.hero-subtitle { color: #8899cc; font-size: 1.05rem; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SAMPLE DATASET (built-in, no download needed)
# ─────────────────────────────────────────
SPAM_SAMPLES = [
    "Congratulations! You've won $1,000,000! Click here to claim your prize now!",
    "FREE offer! Buy now and get 50% off. Limited time deal. Act fast!",
    "You have been selected for a special reward. Call 1-800-WINNER immediately.",
    "URGENT: Your bank account has been compromised. Verify your details now.",
    "Make money fast! Work from home and earn $5000 per week. No experience needed.",
    "Get cheap Viagra and other medications. No prescription required.",
    "You are a winner in our lottery. Send your details to claim $500,000.",
    "Lose 30 pounds in 30 days! Miracle weight loss pill, no diet needed.",
    "Earn thousands weekly from home. Guaranteed income! Register today for free.",
    "ALERT: Account suspended. Click link to verify and restore access immediately.",
    "Hot singles in your area want to meet you tonight! Click here.",
    "Investment opportunity! Double your money in 48 hours. Guaranteed returns!",
    "Your PayPal account is limited. Confirm your identity to avoid suspension.",
    "Exclusive deal for you only! Buy crypto now before it's too late!",
    "Claim your free iPhone 14. You are our lucky visitor today!",
    "Dear winner, you have been selected to receive a cash prize of $50,000.",
    "FINAL NOTICE: Last chance to claim your inheritance of $1.2 million.",
    "Unlock your credit score for FREE! No credit card needed. Sign up now!",
    "BIG SALE! Designer goods at 90% off. Only today. Shop now.",
    "You owe $0. Click to see your credit report. Act now before it expires.",
    "Refinance your mortgage at 0%! Save thousands starting today.",
    "FREE vacation packages available for selected customers. Claim yours.",
    "Your email won in our e-lottery! Contact us to receive your prize.",
    "Hair loss cure discovered by doctors. Regrow full hair in 2 weeks!",
    "Make $3000 per day from home. Absolutely no skills required.",
    "URGENT REPLY NEEDED: I am a Nigerian prince needing your help.",
    "Click here to unsubscribe or you'll keep receiving these special offers.",
    "Cheap insurance quotes! Get covered for as low as $1/day. Apply now!",
    "You qualify for a government grant of $25,000. No repayment required.",
    "Enlarge your manhood naturally! Doctors won't tell you this secret.",
]

HAM_SAMPLES = [
    "Hi John, are we still meeting for lunch on Thursday? Let me know!",
    "Please find attached the quarterly report for your review and comments.",
    "Reminder: Your dentist appointment is scheduled for Monday at 2 PM.",
    "Hey, just wanted to check in. How did your interview go yesterday?",
    "The team meeting has been rescheduled to 3 PM on Friday. Please update calendars.",
    "Thanks for the birthday wishes! It was a wonderful day with family.",
    "Could you send me the project timeline? I need it for the client presentation.",
    "I saw your LinkedIn post about the conference. Sounds like it was a great event!",
    "Your Amazon order has shipped and will arrive by Wednesday.",
    "Hi, I'm following up on our conversation from last week regarding the proposal.",
    "Can you review the draft document I've attached and share your feedback?",
    "The server maintenance is scheduled for Sunday 2-4 AM. Minimal downtime expected.",
    "Great job on the presentation today! The client was really impressed.",
    "I will be out of office from Dec 24 to Jan 2. Contact Jane for urgent matters.",
    "Just a friendly reminder that rent is due on the 1st. Thanks!",
    "Looking forward to your visit! We have planned a lot of fun activities.",
    "Please confirm your attendance for the annual company picnic by Friday.",
    "I've uploaded the files to the shared drive. Let me know if you can access them.",
    "Congratulations on your promotion! Well deserved after all your hard work.",
    "The new software update has been deployed. Please restart your workstation.",
    "Your flight itinerary for the Chicago trip is attached. Safe travels!",
    "We're planning a surprise party for Sarah next week. Can you make it?",
    "Here are the minutes from today's board meeting. Please review and approve.",
    "Just checking in to see if you received my previous email regarding the contract.",
    "Happy anniversary to you both! Wishing you many more years of happiness.",
    "The book you requested is now available at the library. Pick it up within 7 days.",
    "Could we schedule a call this week to discuss the Q4 marketing strategy?",
    "Your subscription renewal is coming up. No action needed if you wish to continue.",
    "I wanted to share this interesting article I read on machine learning trends.",
    "Thank you for your application. We will be in touch within 5 business days.",
]

# Build dataset
def load_dataset():
    emails = SPAM_SAMPLES + HAM_SAMPLES
    labels = ["spam"] * len(SPAM_SAMPLES) + ["ham"] * len(HAM_SAMPLES)
    df = pd.DataFrame({"email": emails, "label": labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ─────────────────────────────────────────
# TEXT PREPROCESSING
# ─────────────────────────────────────────
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' urllink ', text)
    text = re.sub(r'\$\d+[\,\d]*', ' moneysymbol ', text)
    text = re.sub(r'\b\d+%', ' percentnum ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─────────────────────────────────────────
# FEATURE EXTRACTION (rule-based signals)
# ─────────────────────────────────────────
def extract_features_info(text):
    features = {}
    features['length'] = len(text)
    features['word_count'] = len(text.split())
    features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    features['exclamation_count'] = text.count('!')
    features['has_url'] = int(bool(re.search(r'http|www|\.com', text, re.I)))
    features['has_money'] = int(bool(re.search(r'\$|\bfree\b|\bwin\b|\bprize\b|\bcash\b', text, re.I)))
    features['urgency_words'] = len(re.findall(r'\burgent\b|\bnow\b|\bact\b|\blimited\b|\bimmediately\b', text, re.I))
    features['spam_words'] = len(re.findall(
        r'\bfree\b|\bwin\b|\bwon\b|\bclaim\b|\bprize\b|\boffer\b|\bdeal\b|\bcongratulations\b|\bguaranteed\b', text, re.I
    ))
    return features


# ─────────────────────────────────────────
# TRAIN ALL MODELS
# ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_models():
    df = load_dataset()
    df['clean'] = df['email'].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean'], df['label'], test_size=0.25, random_state=42, stratify=df['label']
    )

    models = {
        "Naive Bayes": Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ('clf',   MultinomialNB(alpha=0.1))
        ]),
        "Logistic Regression": Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ('clf',   LogisticRegression(max_iter=500, C=1.0))
        ]),
        "Linear SVM": Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ('clf',   LinearSVC(max_iter=1000, C=1.0))
        ]),
    }

    results = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred, labels=["spam", "ham"])
        results[name] = {
            "pipeline": pipe,
            "accuracy": acc,
            "report": report,
            "cm": cm,
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
        }

    return results, df, X_test.tolist(), y_test.tolist()


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Spam Detector")
    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    selected_model = st.selectbox(
        "Choose ML Model",
        ["Naive Bayes", "Logistic Regression", "Linear SVM"],
        index=1,
        help="Select the machine learning model for classification."
    )

    confidence_threshold = st.slider(
        "Confidence Threshold", 0.5, 0.99, 0.75, 0.01,
        help="Minimum confidence to flag as spam."
    )

    st.markdown("---")
    st.markdown("### 📋 About")
    st.markdown("""
    This app uses **NLP + Machine Learning** to detect spam emails in real-time.

    **Pipeline:**
    - Text Preprocessing
    - TF-IDF Vectorization
    - ML Classification
    - Confidence Scoring
    """)
    st.markdown("---")
    st.caption("Built with Streamlit · scikit-learn")


# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<h1>🛡️ Smart Email Spam Detector</h1>
<p class="hero-subtitle">Real-time email classification powered by Machine Learning & NLP</p>
""", unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────
# TRAIN ON LOAD
# ─────────────────────────────────────────
with st.spinner("🔄 Training models on email dataset..."):
    model_results, df, X_test_samples, y_test_labels = train_models()

active_model = model_results[selected_model]["pipeline"]
active_report = model_results[selected_model]["report"]
active_acc = model_results[selected_model]["accuracy"]


# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Detect Spam", "📊 Model Performance", "📈 Dataset Analysis", "📚 How It Works"
])


# ═══════════════════════════════════════════
# TAB 1 — DETECT SPAM
# ═══════════════════════════════════════════
with tab1:
    st.markdown("### ✉️ Paste your email below to analyze")

    col_input, col_examples = st.columns([2, 1])

    with col_examples:
        st.markdown("**Quick Examples:**")
        ex_spam = st.button("🔴 Load Spam Example")
        ex_ham  = st.button("🟢 Load Ham Example")

    with col_input:
        default_text = ""
        if ex_spam:
            default_text = "CONGRATULATIONS! You've been selected to win $1,000,000! Click here NOW to claim your prize before it expires. Limited time offer!"
        elif ex_ham:
            default_text = "Hi Sarah, just wanted to confirm our meeting is still on for Thursday at 3 PM. Please let me know if you need to reschedule. Thanks!"

        email_input = st.text_area(
            "Email Content",
            value=default_text,
            height=180,
            placeholder="Paste the full email content here...",
            label_visibility="collapsed"
        )

    analyze_col, clear_col = st.columns([1, 5])
    with analyze_col:
        analyze_btn = st.button("🔍 Analyze Email", use_container_width=True)

    if analyze_btn and email_input.strip():
        with st.spinner("Analyzing..."):
            time.sleep(0.5)

        clean = preprocess_text(email_input)
        prediction = active_model.predict([clean])[0]
        feat_info = extract_features_info(email_input)

        # Confidence
        try:
            proba = active_model.predict_proba([clean])[0]
            spam_idx = list(active_model.classes_).index("spam")
            spam_conf = proba[spam_idx]
        except Exception:
            spam_conf = 0.9 if prediction == "spam" else 0.1

        st.markdown("---")
        st.markdown("### 🎯 Detection Result")

        r1, r2, r3 = st.columns([1, 2, 1])
        with r2:
            if prediction == "spam":
                st.markdown(f"""
                <div class="spam-box">
                    <div class="result-emoji">🚨</div>
                    <div class="result-title" style="color:#e05555;">SPAM DETECTED</div>
                    <div class="result-conf">Spam Confidence: <b>{spam_conf:.1%}</b></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="ham-box">
                    <div class="result-emoji">✅</div>
                    <div class="result-title" style="color:#44cc66;">LEGITIMATE EMAIL</div>
                    <div class="result-conf">Ham Confidence: <b>{1 - spam_conf:.1%}</b></div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🔎 Email Feature Analysis")

        m1, m2, m3, m4 = st.columns(4)
        metrics = [
            (m1, feat_info['word_count'], "Word Count"),
            (m2, f"{feat_info['uppercase_ratio']:.0%}", "Uppercase Ratio"),
            (m3, feat_info['exclamation_count'], "Exclamations"),
            (m4, feat_info['spam_words'], "Spam Keywords"),
        ]
        for col, val, label in metrics:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        flags = [
            (f1, "🔗 Contains URL", feat_info['has_url']),
            (f2, "💰 Money/Prize Words", feat_info['has_money']),
            (f3, "⚡ Urgency Words", feat_info['urgency_words'] > 0),
        ]
        for col, label, present in flags:
            with col:
                icon = "🔴" if present else "🟢"
                status = "Detected" if present else "Not Found"
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:1.6rem">{icon}</div>
                    <div class="metric-label">{label}</div>
                    <div style="color:#8899cc;font-size:0.8rem;margin-top:4px">{status}</div>
                </div>""", unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=spam_conf * 100,
            title={'text': "Spam Probability", 'font': {'color': '#a5b8ff', 'size': 16}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#a5b8ff'},
                'bar': {'color': "#e05555" if prediction == "spam" else "#44cc66"},
                'steps': [
                    {'range': [0, 40],  'color': '#0a2d0a'},
                    {'range': [40, 70], 'color': '#2d2d0a'},
                    {'range': [70, 100],'color': '#2d0a0a'},
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 3},
                    'thickness': 0.8,
                    'value': confidence_threshold * 100
                }
            },
            number={'suffix': '%', 'font': {'color': '#7c9dff', 'size': 36}}
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#a5b8ff',
            height=280,
            margin=dict(t=50, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    elif analyze_btn:
        st.warning("⚠️ Please paste some email content before analyzing.")


# ═══════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Model Performance Comparison")

    # Accuracy comparison bar chart
    model_names = list(model_results.keys())
    accuracies = [model_results[m]["accuracy"] * 100 for m in model_names]
    precisions = [model_results[m]["report"]["spam"]["precision"] * 100 for m in model_names]
    recalls    = [model_results[m]["report"]["spam"]["recall"] * 100 for m in model_names]
    f1s        = [model_results[m]["report"]["spam"]["f1-score"] * 100 for m in model_names]

    fig_bar = go.Figure(data=[
        go.Bar(name='Accuracy',  x=model_names, y=accuracies,  marker_color='#7c9dff'),
        go.Bar(name='Precision', x=model_names, y=precisions,  marker_color='#44cc66'),
        go.Bar(name='Recall',    x=model_names, y=recalls,     marker_color='#ffaa44'),
        go.Bar(name='F1 Score',  x=model_names, y=f1s,         marker_color='#e05555'),
    ])
    fig_bar.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,15,30,0.8)',
        font_color='#a5b8ff',
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        yaxis=dict(title='Score (%)', range=[0, 110], gridcolor='#1e2a4a'),
        xaxis=dict(gridcolor='#1e2a4a'),
        height=400,
        margin=dict(t=30)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown(f"### 🎯 Active Model: **{selected_model}**")

    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, val, label in [
        (mc1, f"{active_acc:.1%}", "Accuracy"),
        (mc2, f"{active_report['spam']['precision']:.1%}", "Precision"),
        (mc3, f"{active_report['spam']['recall']:.1%}", "Recall"),
        (mc4, f"{active_report['spam']['f1-score']:.1%}", "F1 Score"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Confusion matrix
    cm = model_results[selected_model]["cm"]
    fig_cm = go.Figure(go.Heatmap(
        z=cm,
        x=["Predicted Spam", "Predicted Ham"],
        y=["Actual Spam", "Actual Ham"],
        colorscale=[[0,'#0a0e1f'],[0.5,'#1e3a8a'],[1,'#7c9dff']],
        text=cm, texttemplate="%{text}",
        textfont={"size": 22, "color": "white"},
        showscale=False,
    ))
    fig_cm.update_layout(
        title=dict(text="Confusion Matrix", font=dict(color='#a5b8ff')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,15,30,0.8)',
        font_color='#a5b8ff',
        height=340,
        margin=dict(t=50)
    )
    st.plotly_chart(fig_cm, use_container_width=True)


# ═══════════════════════════════════════════
# TAB 3 — DATASET ANALYSIS
# ═══════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Dataset Overview")

    da1, da2, da3 = st.columns(3)
    with da1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Total Emails</div>
        </div>""", unsafe_allow_html=True)
    with da2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#e05555">{(df['label']=='spam').sum()}</div>
            <div class="metric-label">Spam Emails</div>
        </div>""", unsafe_allow_html=True)
    with da3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#44cc66">{(df['label']=='ham').sum()}</div>
            <div class="metric-label">Ham Emails</div>
        </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        counts = df['label'].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=["Spam", "Ham"],
            values=[counts.get("spam", 0), counts.get("ham", 0)],
            hole=0.55,
            marker=dict(colors=["#e05555", "#44cc66"]),
            textfont=dict(color="white"),
        ))
        fig_pie.update_layout(
            title=dict(text="Class Distribution", font=dict(color='#a5b8ff')),
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#a5b8ff',
            showlegend=True,
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            height=340,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        df['word_count'] = df['email'].apply(lambda x: len(x.split()))
        spam_wc = df[df['label'] == 'spam']['word_count']
        ham_wc  = df[df['label'] == 'ham']['word_count']

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=spam_wc, name='Spam', marker_color='#e05555', opacity=0.75, nbinsx=15))
        fig_hist.add_trace(go.Histogram(x=ham_wc,  name='Ham',  marker_color='#44cc66', opacity=0.75, nbinsx=15))
        fig_hist.update_layout(
            barmode='overlay',
            title=dict(text="Word Count Distribution", font=dict(color='#a5b8ff')),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,15,30,0.8)',
            font_color='#a5b8ff',
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            xaxis=dict(title='Word Count', gridcolor='#1e2a4a'),
            yaxis=dict(title='Frequency',  gridcolor='#1e2a4a'),
            height=340,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Sample dataset table
    st.markdown("### 📋 Sample Emails")
    show_df = df[['label', 'email']].head(12).copy()
    show_df['label'] = show_df['label'].apply(
        lambda l: f"🔴 SPAM" if l == "spam" else "🟢 HAM"
    )
    st.dataframe(
        show_df.rename(columns={"label": "Label", "email": "Email Content"}),
        use_container_width=True,
        height=380,
    )


# ═══════════════════════════════════════════
# TAB 4 — HOW IT WORKS
# ═══════════════════════════════════════════
with tab4:
    st.markdown("### 📚 How the Spam Detector Works")

    st.markdown("""
    This application uses a classic **NLP + Machine Learning pipeline** to classify emails as spam or legitimate (ham).
    """)

    steps = [
        ("1️⃣", "Text Preprocessing",
         "Raw email text is cleaned: lowercased, URLs replaced with tokens, money patterns flagged, punctuation removed, and whitespace normalized."),
        ("2️⃣", "TF-IDF Vectorization",
         "Text is converted to numerical features using **Term Frequency–Inverse Document Frequency (TF-IDF)** with 1-gram and 2-gram tokens. This captures both single words and word pairs."),
        ("3️⃣", "ML Classification",
         "Three models are trained and available: **Naive Bayes** (probabilistic, fast), **Logistic Regression** (linear boundary, interpretable), and **Linear SVM** (maximum margin classifier, robust)."),
        ("4️⃣", "Confidence Scoring",
         "For probabilistic models (NB, LR), a spam probability score is computed. The configurable threshold determines the final classification boundary."),
        ("5️⃣", "Feature Analysis",
         "Rule-based signals augment ML predictions: uppercase ratio, exclamation marks, spam keywords, URL presence, and urgency language."),
    ]

    for icon, title, desc in steps:
        st.markdown(f"""
        <div class="metric-card" style="text-align:left; margin:10px 0;">
            <span style="font-size:1.4rem">{icon}</span>
            <span style="font-size:1.05rem; font-weight:600; color:#7c9dff; margin-left:10px">{title}</span>
            <p style="color:#c4cfea; margin:8px 0 0 0; font-size:0.92rem">{desc}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧠 Model Comparison")
    comparison = pd.DataFrame({
        "Model": ["Naive Bayes", "Logistic Regression", "Linear SVM"],
        "Type": ["Probabilistic", "Linear", "Margin-based"],
        "Speed": ["⚡⚡⚡ Very Fast", "⚡⚡ Fast", "⚡⚡ Fast"],
        "Interpretability": ["🟢 High", "🟢 High", "🟡 Medium"],
        "Best For": ["Baseline, real-time", "General purpose", "High accuracy"],
        "Gives Probability": ["✅ Yes", "✅ Yes", "❌ No"],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🔑 Key Spam Signals Detected")
    signals = [
        "💰 Money/prize/lottery mentions",
        "🔗 Suspicious URLs or links",
        "⚡ Urgency language (act now, limited time)",
        "🔠 Excessive uppercase letters",
        "❗ Multiple exclamation marks",
        "🎁 Free offer / no cost claims",
        "🏦 Fake account alerts / phishing",
        "💊 Medication / health miracle claims",
    ]
    cols = st.columns(2)
    for i, s in enumerate(signals):
        with cols[i % 2]:
            st.markdown(f"""<div class="metric-card" style="text-align:left;padding:12px 16px;">
                <span style="font-size:0.92rem;color:#c4cfea">{s}</span>
            </div>""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#4a5580; font-size:0.82rem; padding:16px 0;">
    🛡️ Smart Email Spam Detector · Built with Streamlit & scikit-learn · 
    <span style="font-family:'Space Mono',monospace">NLP + ML Pipeline</span>
</div>
""", unsafe_allow_html=True)
