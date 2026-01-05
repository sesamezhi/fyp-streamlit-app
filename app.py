# =========================================================
# Streamlit App — Review Sentiment & Type Analysis
# Binary & 3-Class Sentiment (Imbalanced BERT)
# Split Probability Charts + 6 Word Clouds
# Simple Explainability (Indicative Words)
# =========================================================

import streamlit as st
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from transformers import BertTokenizerFast, BertForSequenceClassification
from wordcloud import WordCloud

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Review Analysis Dashboard", layout="wide")

st.title("📊 Review Sentiment & Type Analysis Dashboard")
st.caption("Final Year Project by **Lim Dao Zhi (TP074115)**")

st.write(
    """
    This application analyses customer reviews using **BERT-based deep learning models**
    to predict:
    - **Sentiment** (Binary or 3-Class)
    - **Review Type** (Apps, Products, Services)

    The system is trained on **51,718 reviews aggregated from Amazon, Google Play Store,
    and Yelp**, covering multiple domains and review styles.
    """
)

# -------------------------
# Sidebar — Model Selection
# -------------------------
st.sidebar.header("⚙️ Model Settings")

sentiment_mode = st.sidebar.radio(
    "Select Sentiment Model",
    ["Binary (Positive / Negative)", "3-Class (Negative / Neutral / Positive)"]
)

show_wordclouds = st.sidebar.checkbox("Show Word Clouds", value=True)

# -------------------------
# Load Models
# -------------------------
@st.cache_resource
def load_models(mode):
    if mode.startswith("Binary"):
        sentiment_repo = "limdaozhi/bert-imbalanced-sentiment-binary"
        sent_labels = ["Negative", "Positive"]
    else:
        sentiment_repo = "limdaozhi/bert-imbalanced-sentiment-3class"
        sent_labels = ["Negative", "Neutral", "Positive"]

    sentiment_model = BertForSequenceClassification.from_pretrained(sentiment_repo)
    type_model = BertForSequenceClassification.from_pretrained(
        "limdaozhi/bert-review-type-model"
    )
    tokenizer = BertTokenizerFast.from_pretrained(sentiment_repo)

    return sentiment_model, type_model, tokenizer, sent_labels


try:
    sentiment_model, type_model, tokenizer, sent_labels = load_models(sentiment_mode)
    sentiment_model.eval()
    type_model.eval()
except Exception as e:
    st.error("❌ Failed to load models.")
    st.exception(e)
    st.stop()

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("imbalance_reviews.csv")

# -------------------------
# User Input
# -------------------------
st.subheader("📝 Enter a Review")
user_input = st.text_area("Type or paste a review below:", height=120)

if st.button("Analyze Review") and user_input.strip():

    inputs = tokenizer(
        user_input,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        sent_logits = sentiment_model(**inputs).logits
        type_logits = type_model(**inputs).logits

    # -------------------------
    # Predictions
    # -------------------------
    sent_probs = torch.softmax(sent_logits, dim=1)[0].numpy()
    type_probs = torch.softmax(type_logits, dim=1)[0].numpy()

    sentiment = sent_labels[int(np.argmax(sent_probs))]
    sentiment_conf = sent_probs.max() * 100

    type_labels = ["Apps", "Products", "Services"]
    review_type = type_labels[int(np.argmax(type_probs))]

    # -------------------------
    # Results Display
    # -------------------------
    st.subheader("🔍 Prediction Results")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Sentiment", sentiment)
        st.progress(float(sentiment_conf / 100))
        st.write(f"Confidence: **{sentiment_conf:.2f}%**")

    with col2:
        st.metric("Predicted Review Type", review_type)

    # =====================================================
    # PREDICTION PROBABILITIES — PIE CHARTS
    # =====================================================
    st.subheader("📈 Prediction Probabilities")

    colA, colB = st.columns(2)

    with colA:
        explode = [0.1 if i == np.argmax(sent_probs) else 0 for i in range(len(sent_probs))]
        fig, ax = plt.subplots()
        ax.pie(sent_probs, labels=sent_labels, autopct="%.1f%%",
               startangle=90, explode=explode)
        ax.set_title("Sentiment Probability Distribution")
        ax.axis("equal")
        st.pyplot(fig)

    with colB:
        explode = [0.1 if i == np.argmax(type_probs) else 0 for i in range(len(type_probs))]
        fig, ax = plt.subplots()
        ax.pie(type_probs, labels=type_labels, autopct="%.1f%%",
               startangle=90, explode=explode)
        ax.set_title("Review Type Probability Distribution")
        ax.axis("equal")
        st.pyplot(fig)

    # =====================================================
    # EXPLAINABILITY
    # =====================================================
    st.subheader("🧠 Indicative Words in the Review")

    tokens = tokenizer.tokenize(user_input.lower())
    keywords = [t.replace("##", "") for t in tokens if t.isalpha()]

    sentiment_keywords = keywords[:10]
    type_keywords = keywords[10:18]

    st.markdown(f"**Why is this a _{sentiment.lower()}_ review?**")
    st.write(", ".join(sentiment_keywords) if sentiment_keywords else "No strong indicators detected.")

    st.markdown(f"**Why is this a _{review_type.lower()}_ review?**")
    st.write(", ".join(type_keywords) if type_keywords else "No strong indicators detected.")

# =========================================================
# WORD CLOUD DASHBOARD
# =========================================================
if show_wordclouds:

    st.subheader("☁️ Word Cloud Dashboard")

    def generate_wordcloud(text, title):
        if not isinstance(text, str) or not text.strip():
            st.write(f"No data for {title}")
            return
        wc = WordCloud(width=400, height=300, background_color="white").generate(text)
        fig, ax = plt.subplots()
        ax.imshow(np.array(wc.to_image()))
        ax.axis("off")
        ax.set_title(title)
        st.pyplot(fig)

    st.markdown("### 🧭 Sentiment Word Clouds")
    col1, col2, col3 = st.columns(3)

    with col1:
        generate_wordcloud(" ".join(df[df["sentiment"].isin(["negative","very negative"])]["cleaned_text"]), "Negative")
    with col2:
        generate_wordcloud(" ".join(df[df["sentiment"]=="neutral"]["cleaned_text"]), "Neutral")
    with col3:
        generate_wordcloud(" ".join(df[df["sentiment"].isin(["positive","very positive"])]["cleaned_text"]), "Positive")

    st.markdown("### ✍️ Review Type Word Clouds")
    col4, col5, col6 = st.columns(3)

    with col4:
        generate_wordcloud(" ".join(df[df["rev_type"]=="apps"]["cleaned_text"]), "Apps")
    with col5:
        generate_wordcloud(" ".join(df[df["rev_type"]=="products"]["cleaned_text"]), "Products")
    with col6:
        generate_wordcloud(" ".join(df[df["rev_type"]=="services"]["cleaned_text"]), "Services")

# =========================================================
# DATASET CREDITS
# =========================================================
st.markdown("---")
st.subheader("📚 Dataset Sources & Credits")

st.markdown(
    """
All datasets used in this project were sourced from **Kaggle.com** and merged
to form a unified review corpus for analysis.

**Dataset Sources:**
- **Amazon Reviews Dataset**  
  https://www.kaggle.com/datasets/dongrelaxman/amazon-reviews-dataset
- **Google Play Store Reviews**  
  https://www.kaggle.com/datasets/prakharrathi25/google-play-store-reviews
- **Google Play Messenger Reviews (6,000 samples)**  
  https://www.kaggle.com/datasets/trainingdatapro/6000-messengers-reviews-google-play
- **Yelp Restaurant Reviews**  
  https://www.kaggle.com/datasets/farukalam/yelp-restaurant-reviews

These datasets collectively contributed to the **51,718 review records**
used for training and evaluation.
"""
)

st.success("✅ Sentiment Analysis System Ready")
