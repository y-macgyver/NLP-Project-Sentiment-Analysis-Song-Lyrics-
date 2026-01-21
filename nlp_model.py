# =========================================
# STREAMLIT SENTIMENT ANALYSIS DASHBOARD
# SONG LYRICS NLP PROJECT
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

st.title("🎵 Song Lyrics Sentiment Analysis Dashboard")

# =========================================
# NLTK SETUP
# =========================================
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# =========================================
# FILE UPLOAD
# =========================================
st.sidebar.header("📂 Upload Dataset")
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
st.sidebar.subheader("⚙️ Dataset Settings")

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
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

df['clean_lyrics'] = df[text_column].apply(clean_text)

# =========================================
# LOAD MODELS (CACHED)
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
# SENTIMENT PREDICTION
# =========================================
def get_sentiment(text):
    result = sentiment_model(text[:512])[0]
    return result['label'], result['score']

# =========================================
# EMOTION PREDICTION
# =========================================
def get_emotions(text):
    emotions = emotion_model(text[:512])[0]
    return {e['label']: e['score'] for e in emotions}

# =========================================
# ANALYZE BUTTON
# =========================================
if st.button("🔍 Analyze Lyrics Dataset"):
    with st.spinner("Analyzing lyrics using Transformer models..."):
        df[['sentiment', 'sentiment_confidence']] = df['clean_lyrics'].apply(
            lambda x: pd.Series(get_sentiment(x))
        )

        emotion_df = df['clean_lyrics'].apply(get_emotions).apply(pd.Series)
        df_final = pd.concat([df, emotion_df], axis=1)

    st.success("Analysis completed!")

    # =====================================
    # SENTIMENT DISTRIBUTION
    # =====================================
    st.subheader("📊 Sentiment Polarity Distribution")

    sentiment_counts = df_final['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']

    fig_sentiment = px.pie(
        sentiment_counts,
        names='Sentiment',
        values='Count'
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)

    # =====================================
    # EMOTION ANALYSIS
    # =====================================
    st.subheader("🎭 Emotion Analysis")

    emotion_columns = emotion_df.columns
    emotion_avg = df_final[emotion_columns].mean().reset_index()
    emotion_avg.columns = ['Emotion', 'Score']

    fig_emotion = px.bar(
        emotion_avg,
        x='Emotion',
        y='Score'
    )
    st.plotly_chart(fig_emotion, use_container_width=True)

    # =====================================
    # SAMPLE RESULTS TABLE
    # =====================================
    st.subheader("📝 Sample Lyrics Analysis")
    st.dataframe(
        df_final[[text_column, 'sentiment', 'sentiment_confidence']]
        .head(10),
        use_container_width=True
    )

# =========================================
# SINGLE LYRIC TESTING
# =========================================
st.markdown("---")
st.subheader("🎧 Test Custom Lyrics")

user_input = st.text_area(
    "Enter song lyrics:",
    height=150
)

if st.button("Analyze Lyrics"):
    if user_input.strip():
        sentiment, confidence = get_sentiment(user_input)
        emotions = get_emotions(user_input)

        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence:.2f}")

        emotion_df_user = pd.DataFrame(
            emotions.items(),
            columns=['Emotion', 'Score']
        )

        fig_user_emotion = px.bar(
            emotion_df_user,
            x='Emotion',
            y='Score',
            title="Emotion Breakdown"
        )

        st.plotly_chart(fig_user_emotion, use_container_width=True)
    else:
        st.warning("Please enter some lyrics.")
