# =========================================
# STREAMLIT SENTIMENT & EMOTION DASHBOARD
# SONG LYRICS NLP PROJECT
# =========================================

import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from transformers import pipeline
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Song Lyrics Sentiment Analysis",
    layout="wide"
)

st.title("🎵 Song Lyrics Sentiment & Emotion Analysis Dashboard")

# =========================================
# NLTK SETUP
# =========================================
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# =========================================
# FILE UPLOAD
# =========================================
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file containing song lyrics",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to begin analysis.")
    st.stop()

# =========================================
# LOAD DATASET
# =========================================
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

df = load_data(uploaded_file)

# =========================================
# COLUMN SELECTION
# =========================================
st.sidebar.subheader("Dataset Settings")

text_column = st.sidebar.selectbox(
    "Select lyrics text column",
    df.columns
)

# =========================================
# TEXT PREPROCESSING
# =========================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join(w for w in text.split() if w not in stop_words)
    return text

# Keep RAW + CLEAN text
df["raw_lyrics"] = df[text_column].astype(str)
df["clean_lyrics"] = df[text_column].apply(clean_text)

# Remove empty rows
df = df[df["clean_lyrics"].str.strip() != ""]

# =========================================
# LOAD MODELS
# =========================================
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

sentiment_model = load_sentiment_model()
emotion_model = load_emotion_model()

# =========================================
# SENTIMENT FUNCTION
# =========================================
def get_sentiment(text):
    try:
        result = sentiment_model(text[:512])[0]
        return result["label"], float(result["score"])
    except Exception:
        return "NEUTRAL", 0.0

# =========================================
# EMOTION FUNCTION (ROBUST)
# =========================================
def get_emotions(text):
    labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
    default = dict.fromkeys(labels, 0.0)

    if not isinstance(text, str) or len(text.strip()) < 20:
        return default

    try:
        output = emotion_model(text[:512])
        if not output or not output[0]:
            return default
        return {e["label"]: float(e["score"]) for e in output[0]}
    except Exception:
        return default

# =========================================
# ANALYZE BUTTON
# =========================================
if st.button("Analyze Lyrics Dataset"):
    with st.spinner("Analyzing lyrics using Transformer models..."):

        # Sentiment
        df[["sentiment", "sentiment_confidence"]] = df["clean_lyrics"].apply(
            lambda x: pd.Series(get_sentiment(x))
        )

        # Emotion (RAW lyrics)
        emotion_results = df["raw_lyrics"].apply(get_emotions)
        emotion_df = pd.DataFrame(list(emotion_results))

        df_final = pd.concat([df, emotion_df], axis=1)

    st.success("Analysis completed!")

    # =====================================
    # SENTIMENT DISTRIBUTION
    # =====================================
    st.subheader("Sentiment Polarity Distribution")

    sentiment_counts = df_final["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    fig_sentiment = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Count"
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)


    # =====================================
    # EMOTION ANALYSIS (Word Cloud)
    # =====================================
    st.subheader("Emotion Analysis (Word Cloud)")
    
    emotion_columns = emotion_df.columns.tolist()
    emotion_avg = df_final[emotion_columns].mean()
    
    # Convert to dict for word cloud
    emotion_dict = emotion_avg.to_dict()
    
    # Generate word cloud
    wordcloud = WordCloud(
        background_color="white",
        colormap="coolwarm",
        width=800,
        height=400
    ).generate_from_frequencies(emotion_dict)
    
    # Display with matplotlib and streamlit
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


    # =====================================
    # SAMPLE TABLE
    # =====================================
    st.subheader("Sample Lyrics Analysis")

    st.dataframe(
        df_final[[text_column, "sentiment", "sentiment_confidence"]].head(50),
        use_container_width=True
    )

# =========================================
# SINGLE LYRIC TESTING
# =========================================
st.markdown("---")
st.subheader("Test Custom Lyrics")

user_input = st.text_area(
    "Enter song lyrics:",
    height=150
)

if st.button("Analyze Lyrics"):
    if user_input.strip():
        sentiment, confidence = get_sentiment(clean_text(user_input))
        emotions = get_emotions(user_input)

        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence:.2f}")

        emotion_df_user = pd.DataFrame(
            emotions.items(),
            columns=["Emotion", "Score"]
        )

        wordcloud = WordCloud(
            background_color="white",
            colormap="coolwarm",
            width=800,
            height=400
        ).generate_from_frequencies(emotion_dict)
        
        # Display with matplotlib and streamlit
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.warning("Please enter some lyrics.")
