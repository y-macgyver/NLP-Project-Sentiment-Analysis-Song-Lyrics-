import streamlit as st
import pandas as pd
import plotly.express as px
from nlp_model import analyze_text

st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    layout="wide"
)

st.title("🎵 Sentiment Analysis Dashboard (Lyrics / Social Media)")

st.write(
    "Analyze sentiment polarity and emotions using Transformer-based NLP models."
)

# Session state to store results
if "results" not in st.session_state:
    st.session_state.results = []

text_input = st.text_area(
    "Enter song lyrics or social media text:",
    height=150
)

if st.button("Analyze"):
    if text_input.strip():
        result = analyze_text(text_input)
        st.session_state.results.append(result)

        st.success("Analysis Completed")

        st.markdown(
            f"""
            **Sentiment:** {result['sentiment']}  
            **Sentiment Score:** {result['sentiment_score']}  
            **Emotion:** {result['emotion']}  
            **Emotion Confidence:** {result['emotion_score']}
            """
        )
    else:
        st.warning("Please enter some text.")

# Visualization
if st.session_state.results:
    df = pd.DataFrame(st.session_state.results)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Polarity Distribution")
        fig1 = px.histogram(
            df,
            x="sentiment",
            title="Sentiment Distribution"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Emotion Distribution")
        fig2 = px.histogram(
            df,
            x="emotion",
            title="Emotion Distribution"
        )
        st.plotly_chart(fig2, use_container_width=True)
