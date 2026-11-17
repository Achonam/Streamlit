
import streamlit as st
from transformers import pipeline


st.set_page_config(page_tittle="Mely AI App")
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
    st.write("Type as snetence")
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
    st.wriet(f"Type a Sentence")
    text = st.text_area("Your text", "I love Learning AI")
    
    
    if st.button("Predict word"):
        if not text.strip