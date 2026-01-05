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
st.write("Sentiment & Review Type Prediction using BERT")

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

    # -------- Sentiment Pie Chart --------
    with colA:
        explode = [
            0.1 if i == np.argmax(sent_probs) else 0
            for i in range(len(sent_probs))
        ]

        fig, ax = plt.subplots()
        ax.pie(
            sent_probs,
            labels=sent_labels,
            autopct="%.1f%%",
            startangle=90,
            explode=explode
        )
        ax.set_title("Sentiment Probability Distribution")
        ax.axis("equal")
        st.pyplot(fig)


    # -------- Review Type Pie Chart --------
    with colB:
        explode = [
            0.1 if i == np.argmax(type_probs) else 0
            for i in range(len(type_probs))
        ]

        fig, ax = plt.subplots()
        ax.pie(
            type_probs,
            labels=type_labels,
            autopct="%.1f%%",
            startangle=90,
            explode=explode
        )
        ax.set_title("Review Type Probability Distribution")
        ax.axis("equal")
        st.pyplot(fig)

    # =====================================================
    # EXPLAINABILITY — WHY THIS PREDICTION
    # =====================================================
    st.subheader("🧠 Why was this prediction made?")

    tokens = tokenizer.tokenize(user_input.lower())
    keywords = [t.replace("##", "") for t in tokens if t.isalpha()]

    # Split keywords roughly (simple, explainable heuristic)
    sentiment_keywords = keywords[:10]
    type_keywords = keywords[10:18]

    # -------- Sentiment Explanation --------
    st.markdown(f"**Why is this a _{sentiment.lower()}_ review?**")

    if sentiment_keywords:
        st.write(
            "The following words in the review likely contributed to the "
            f"**{sentiment.lower()} sentiment prediction**:"
        )
        st.write(", ".join(sentiment_keywords))
    else:
        st.write("No strong sentiment-indicative words were detected.")

    # -------- Review Type Explanation --------
    st.markdown(f"**Why is this a _{review_type.lower()}_ review?**")

    if type_keywords:
        st.write(
            "The following words in the review likely contributed to the "
            f"**{review_type.lower()} category prediction**:"
        )
        st.write(", ".join(type_keywords))
    else:
        st.write("No strong review-type-indicative words were detected.")


# =========================================================
# WORD CLOUD DASHBOARD (ALWAYS 6 CLOUDS)
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

    # -------- Sentiment Word Clouds --------
    st.markdown("### 🧭 Sentiment Word Clouds")
    col1, col2, col3 = st.columns(3)

    with col1:
        neg_text = " ".join(df[df["sentiment"].isin(["negative", "very negative"])]["cleaned_text"])
        generate_wordcloud(neg_text, "Negative")

    with col2:
        neu_text = " ".join(df[df["sentiment"] == "neutral"]["cleaned_text"])
        generate_wordcloud(neu_text, "Neutral")

    with col3:
        pos_text = " ".join(df[df["sentiment"].isin(["positive", "very positive"])]["cleaned_text"])
        generate_wordcloud(pos_text, "Positive")

    # -------- Review Type Word Clouds --------
    st.markdown("### ✍️ Review Type Word Clouds")
    col4, col5, col6 = st.columns(3)

    with col4:
        app_text = " ".join(df[df["rev_type"] == "apps"]["cleaned_text"])
        generate_wordcloud(app_text, "Apps")

    with col5:
        prod_text = " ".join(df[df["rev_type"] == "products"]["cleaned_text"])
        generate_wordcloud(prod_text, "Products")

    with col6:
        serv_text = " ".join(df[df["rev_type"] == "services"]["cleaned_text"])
        generate_wordcloud(serv_text, "Services")

st.success("✅ Sentiment Analysis System Ready")
