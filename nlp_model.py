# =========================================
# STREAMLIT SENTIMENT & EMOTION DASHBOARD
# SONG LYRICS NLP PROJECT (FULL FIXED VERSION)
# =========================================

import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from transformers import pipeline
import plotly.express as px

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

df["raw_lyrics"] = df[text_column].astype(str)
df["clean_lyrics"] = df[text_column].apply(clean_text)

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

emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

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
# EMOTION FUNCTION
# =========================================
def get_emotions(text):
    default = dict.fromkeys(emotion_labels, 0.0)

    if not isinstance(text, str) or len(text.strip()) < 20:
        return default

    try:
        output = emotion_model(text[:512])
        if not output or not output[0]:
            return default

        scores = {e["label"]: float(e["score"]) for e in output[0]}

        # Ensure all emotions exist
        for e in emotion_labels:
            scores.setdefault(e, 0.0)

        return scores

    except Exception:
        return default

# =========================================
# ANALYZE BUTTON
# =========================================
if st.button("Analyze Lyrics Dataset"):
    with st.spinner("Analyzing lyrics using Transformer models..."):

        df[["sentiment", "sentiment_confidence"]] = df["clean_lyrics"].apply(
            lambda x: pd.Series(get_sentiment(x))
        )

        emotion_results = df["raw_lyrics"].apply(get_emotions)
        emotion_df = pd.DataFrame(list(emotion_results))

        df_final = pd.concat([df, emotion_df], axis=1)

        df_final[emotion_labels] = df_final[emotion_labels].fillna(0)

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
    # EMOTION PIE (DATASET)
    # =====================================
    st.subheader("Emotion Analysis (Average Scores)")

    emotion_avg = df_final[emotion_labels].mean().reset_index()
    emotion_avg.columns = ["Emotion", "Score"]

    if emotion_avg["Score"].sum() == 0:
        st.warning("No emotion data detected.")
    else:
        fig_emotion = px.pie(
            emotion_avg,
            names="Emotion",
            values="Score",
            title="Emotion Analysis (Average Scores)"
        )

        st.plotly_chart(fig_emotion, use_container_width=True)

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
        ).fillna(0)

        if emotion_df_user["Score"].sum() == 0:
            st.warning("Emotion model returned empty scores.")
        else:
            fig_user_emotion = px.pie(
                emotion_df_user,
                names="Emotion",
                values="Score",
                title="Emotion Breakdown"
            )

            st.plotly_chart(fig_user_emotion, use_container_width=True)

    else:
        st.warning("Please enter some lyrics.")
