# =========================================================
# Streamlit App — Review Sentiment & Type Analysis
# Uses Model 4B (Binary BERT, Imbalanced Dataset)
# FIXED WordCloud Rendering
# =========================================================

import streamlit as st
import torch
import pandas as pd
import matplotlib.pyplot as plt

from transformers import BertTokenizerFast, BertForSequenceClassification
from wordcloud import WordCloud

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Review Analysis Dashboard",
    layout="wide"
)

st.title("📊 Review Sentiment & Type Analysis Dashboard")
st.write("Binary Sentiment + Review Type Prediction using BERT")

# -------------------------
# Load Models & Tokenizer
# -------------------------
@st.cache_resource
def load_models():
    sentiment_model = BertForSequenceClassification.from_pretrained(
        "bert_imbalanced_sentiment_binary"
    )
    type_model = BertForSequenceClassification.from_pretrained(
        "bert_review_type_model"
    )
    tokenizer = BertTokenizerFast.from_pretrained(
        "bert_imbalanced_sentiment_binary"
    )
    return sentiment_model, type_model, tokenizer

sentiment_model, type_model, tokenizer = load_models()

sentiment_model.eval()
type_model.eval()

# -------------------------
# Load Dataset for Dashboard
# -------------------------
df = pd.read_csv("imbalance_reviews.csv")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("📌 Options")
show_wordclouds = st.sidebar.checkbox("Show Word Clouds", value=True)

# -------------------------
# User Input Section
# -------------------------
st.subheader("📝 Enter a Review")
user_input = st.text_area(
    "Type or paste a review below:",
    height=120
)

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
    # Sentiment Prediction
    # -------------------------
    sent_probs = torch.softmax(sent_logits, dim=1)[0]
    sent_labels = ["Negative", "Positive"]

    sentiment = sent_labels[torch.argmax(sent_probs).item()]
    sentiment_conf = torch.max(sent_probs).item()

    # -------------------------
    # Review Type Prediction
    # -------------------------
    type_probs = torch.softmax(type_logits, dim=1)[0]
    type_labels = ["Apps", "Products", "Services"]

    review_type = type_labels[torch.argmax(type_probs).item()]

    # -------------------------
    # Display Results
    # -------------------------
    st.subheader("🔍 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Sentiment", sentiment)
        st.progress(sentiment_conf)
        st.write(f"Confidence: **{sentiment_conf:.2f}**")

    with col2:
        st.metric("Predicted Review Type", review_type)

    st.subheader("📈 Prediction Probabilities")

    prob_df = pd.DataFrame({
        "Class": sent_labels + type_labels,
        "Probability": list(sent_probs.numpy()) + list(type_probs.numpy())
    })

    st.bar_chart(prob_df.set_index("Class"))

# =========================================================
# WORD CLOUD DASHBOARD (FIXED)
# =========================================================
if show_wordclouds:

    st.subheader("☁️ Word Cloud Dashboard")

    def generate_wordcloud(text, title):
        if not isinstance(text, str) or not text.strip():
            st.write(f"No data available for {title}")
            return

        wc = WordCloud(
            width=400,
            height=300,
            background_color="white",
            max_words=100
        ).generate(text)

        fig, ax = plt.subplots()
        ax.imshow(wc.to_array())   # ✅ FIXED
        ax.axis("off")
        ax.set_title(title)
        st.pyplot(fig)

    col1, col2, col3 = st.columns(3)

    # -------------------------
    # Sentiment Word Clouds
    # -------------------------
    with col1:
        neg_text = " ".join(
            df[df["sentiment"].isin(["negative", "very negative"])]["cleaned_text"]
        )
        generate_wordcloud(neg_text, "Negative Reviews")

    with col2:
        pos_text = " ".join(
            df[df["sentiment"].isin(["positive", "very positive"])]["cleaned_text"]
        )
        generate_wordcloud(pos_text, "Positive Reviews")

    # -------------------------
    # Review Type Word Clouds
    # -------------------------
    with col3:
        app_text = " ".join(df[df["rev_type"] == "apps"]["cleaned_text"])
        generate_wordcloud(app_text, "App Reviews")

    col4, col5 = st.columns(2)

    with col4:
        prod_text = " ".join(df[df["rev_type"] == "products"]["cleaned_text"])
        generate_wordcloud(prod_text, "Product Reviews")

    with col5:
        serv_text = " ".join(df[df["rev_type"] == "services"]["cleaned_text"])
        generate_wordcloud(serv_text, "Service Reviews")

st.success("✅ Model 4B Deployed Successfully")
