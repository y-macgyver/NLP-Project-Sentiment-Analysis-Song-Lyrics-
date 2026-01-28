# =========================================
# SONG LYRICS SENTIMENT & EMOTION DASHBOARD
# =========================================

import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
import nltk
import re

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------
st.set_page_config(
    page_title="🎵 Song Lyrics Sentiment & Emotion Analysis",
    layout="wide"
)

st.title("🎵 Song Lyrics Sentiment & Emotion Analysis Dashboard")
st.markdown("Analyze sentiment polarity and emotional tone of song lyrics using NLP and Transformer models.")

# -----------------------------------------
# LOAD DATA
# -----------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("labeled_lyrics_cleaned - 1000 songs.csv")
    return df

df = load_data()

# -----------------------------------------
# SENTIMENT MAPPING
# -----------------------------------------
def map_sentiment(score):
    if score > 0.6:
        return "Positive"
    elif score < 0.4:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["label"].apply(map_sentiment)

# -----------------------------------------
# EMOTION MODEL
# -----------------------------------------
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

emotion_classifier = load_emotion_model()

def get_emotion(text):
    results = emotion_classifier(text[:512])[0]
    return max(results, key=lambda x: x["score"])["label"]

# -----------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------
st.sidebar.header("🔍 Filter Options")

artist = st.sidebar.selectbox(
    "Select Artist",
    ["All"] + sorted(df["artist"].unique())
)

sentiment_filter = st.sidebar.multiselect(
    "Select Sentiment",
    ["Positive", "Neutral", "Negative"],
    default=["Positive", "Neutral", "Negative"]
)

if artist != "All":
    df = df[df["artist"] == artist]

df = df[df["Sentiment"].isin(sentiment_filter)]

# -----------------------------------------
# KPI CARDS
# -----------------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("🎼 Total Songs", len(df))
col2.metric("😊 Positive Songs", (df["Sentiment"] == "Positive").sum())
col3.metric("😢 Negative Songs", (df["Sentiment"] == "Negative").sum())

# -----------------------------------------
# SENTIMENT DISTRIBUTION
# -----------------------------------------
st.subheader("📊 Sentiment Distribution")

sentiment_fig = px.pie(
    df,
    names="Sentiment",
    title="Overall Sentiment Breakdown",
    hole=0.4
)
st.plotly_chart(sentiment_fig, use_container_width=True)

# -----------------------------------------
# EMOTION ANALYSIS
# -----------------------------------------
st.subheader("🎭 Emotion Classification (Sample-Based)")

sample_size = st.slider("Select number of songs to analyze emotion", 10, 200, 50)

sample_df = df.sample(sample_size, random_state=42)
sample_df["Emotion"] = sample_df["lyrics"].apply(get_emotion)

emotion_fig = px.bar(
    sample_df["Emotion"].value_counts().reset_index(),
    x="index",
    y="Emotion",
    labels={"index": "Emotion", "Emotion": "Count"},
    title="Emotion Distribution"
)
st.plotly_chart(emotion_fig, use_container_width=True)

# -----------------------------------------
# SONG-LEVEL ANALYSIS
# -----------------------------------------
st.subheader("🎧 Analyze Individual Song")

selected_song = st.selectbox(
    "Select a song",
    df["song"].unique()
)

song_data = df[df["song"] == selected_song].iloc[0]

st.markdown(f"**Artist:** {song_data['artist']}")
st.markdown(f"**Sentiment:** `{song_data['Sentiment']}`")
st.markdown("**Lyrics:**")
st.text_area("", song_data["lyrics"], height=250)

emotion = get_emotion(song_data["lyrics"])
st.success(f"🎭 Detected Emotion: **{emotion}**")

# -----------------------------------------
# DATA PREVIEW
# -----------------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df[["artist", "song", "Sentiment"]].head(20))
