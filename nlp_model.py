# =========================================
# STREAMLIT SENTIMENT & EMOTION DASHBOARD
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
st.set_page_config(page_title="Song Lyrics Sentiment Analysis", layout="wide")
st.title("🎵 Song Lyrics Sentiment & Emotion Analysis Dashboard")

# =========================================
# NLTK
# =========================================
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# =========================================
# FILE UPLOAD
# =========================================
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file.")
    st.stop()

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

df = load_data(uploaded_file)

text_column = st.sidebar.selectbox("Select lyrics column", df.columns)

# =========================================
# CLEAN TEXT
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
# LOAD MODELS (FIXED)
# =========================================
@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

@st.cache_resource
def load_emotion():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None   # <-- IMPORTANT FIX
    )

sentiment_model = load_sentiment()
emotion_model = load_emotion()

emotion_labels = ["angery", "disgusting", "scare", "happy", "sad", "surprise", "neutral"]

# =========================================
# FUNCTIONS
# =========================================
def get_sentiment(text):
    try:
        r = sentiment_model(text[:512])[0]
        return r["label"], float(r["score"])
    except:
        return "NEUTRAL", 0.0

def get_emotions(text):

    base = dict.fromkeys(emotion_labels, 0.0)

    if not isinstance(text, str) or len(text.strip()) < 20:
        return base

    try:
        result = emotion_model(text[:512])[0]

        scores = {}
        for r in result:
            scores[r["label"].lower()] = float(r["score"])

        for e in emotion_labels:
            scores.setdefault(e, 0.0)

        return scores

    except:
        return base

# =========================================
# ANALYZE DATASET
# =========================================
if st.button("Analyze Lyrics Dataset"):

    with st.spinner("Running models..."):

        df[["sentiment", "sentiment_confidence"]] = df["clean_lyrics"].apply(
            lambda x: pd.Series(get_sentiment(x))
        )

        emotion_results = df["raw_lyrics"].apply(get_emotions)
        emotion_df = pd.DataFrame(list(emotion_results))

        df_final = pd.concat([df, emotion_df], axis=1)
        df_final[emotion_labels] = df_final[emotion_labels].fillna(0)

    st.success("Done!")

    # =====================================
    # SENTIMENT PIE
    # =====================================
    st.subheader("Sentiment Distribution")

    sent_counts = df_final["sentiment"].value_counts().reset_index()
    sent_counts.columns = ["Sentiment", "Count"]

    st.plotly_chart(px.pie(sent_counts, names="Sentiment", values="Count"),
                    use_container_width=True)

    # =====================================
    # EMOTION PIE (DATASET)
    # =====================================
    st.subheader("Emotion Analysis (Average)")

    emotion_avg = df_final[emotion_labels].mean().reset_index()
    emotion_avg.columns = ["Emotion", "Score"]

    if emotion_avg["Score"].sum() == 0:
        st.warning("Emotion model returned empty values.")
    else:
        st.plotly_chart(
            px.pie(emotion_avg, names="Emotion", values="Score"),
            use_container_width=True
        )

    # =====================================
    # SAMPLE
    # =====================================
    st.subheader("Sample Results")

    st.dataframe(df_final[[text_column, "sentiment", "sentiment_confidence"]].head(30),
                 use_container_width=True)

# =========================================
# SINGLE LYRIC
# =========================================
st.markdown("---")
st.subheader("Test Custom Lyrics")

user_input = st.text_area("Enter lyrics:", height=150)

if st.button("Analyze Lyrics"):

    if user_input.strip():

        s, c = get_sentiment(clean_text(user_input))
        emotions = get_emotions(user_input)

        st.write(f"Sentiment: {s}")
        st.write(f"Confidence: {c:.2f}")

        emo_df = pd.DataFrame(emotions.items(), columns=["Emotion", "Score"])

        if emo_df["Score"].sum() == 0:
            st.warning("Emotion model returned empty values.")
        else:
            st.plotly_chart(
                px.pie(emo_df, names="Emotion", values="Score"),
                use_container_width=True
            )

    else:
        st.warning("Please enter lyrics.")
