
import streamlit as st
from transformers import pipeline


st.set_page_config(page_title="Mely AI App")
st.title("DESIGNED TO GUESS WORDS AND SENTIMENTS")



@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    
    
@st.cache_resource
def load_fill_mask_model():
    return pipeline("fill-mask", model="distilbert-base-uncased")
    
    
MASK = "[MASK]"


tab1, tab2 = st.tabs(["Sentiment", "Missing word"])

with tab1:
    st.write("Type as snetence and I will predict if it is POSITIVE or NEGATIVE")
    text = st.text_area("Your sentence", "I Love learning about AI")
    
    if st.button("Check sentiment"):
        if not text.strip():
            st.warning("Please type a sentence first.")
        else:
            with st.spinner("Thinking..."):
                result = load_sentiment_model()(text, truncation=True)[0]
            message: f"{result['label']} (score: {round(result['score'], 3)})"
            if result["label"] == "POSITIVE":
                st.success(message)
            else:
                st.error(message)


with tab2:
    st.write(f"Type a sentence and use {MASK} where a word is missing.")
    sentence = st.text_area("Your sentence", f"we are about {MASK} go home.")
    
    if st.button("Guess the word"):
        if sentence.count(MASK) !=1:
            st.warning(f"Please use {MASK} exactly one time in your sentence.")
        else:
            with st.spinner("Thinking..."):
                guesses = load_fill_mask_model()(sentence, top_k=5)
            st.write("Top 5 guesses:")
            for guess in guesses:
                st.write(f"**{guess['token_str']}** (score: {round(guess['score'], 3)})")
                st.progress(min(float(guess["score"]), 1.0))