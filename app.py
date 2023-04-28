import streamlit as st
import pandas as pd
import re
import pickle
import string
from langdetect import detect
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt')


def clear_text(text):
    # List of all characters and numbers
    eng_chars = string.ascii_letters + string.digits
    # List of all punctuation characters
    punc_chars = string.punctuation
    # Combine the two lists
    remove_chars = eng_chars + punc_chars
    # Remove all characters in the remove_chars list from the text
    text = ''.join(c for c in text if c not in remove_chars)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# All Language Models:

punjabi_lexicon = pd.read_csv("./data/punjabi_lexicon.csv")

marathi_lexicon = pd.read_csv("./data/final_marathi_data.csv")
models_marathi = []
# model_files = ["./models/GBC_marathi.pickle", "./models/LogReg_marathi.pickle",
#                "./models/SupportVec_marathi.pickle", "./models/RandForest_marathi.pickle"]
model_files = ["./models/LogReg_marathi.pickle",
               "./models/SupportVec_marathi.pickle", "./models/RandForest_marathi.pickle", "./models/KNeighbors_marathi.pickle"]
for file in model_files:
    with open(file, "rb") as f:
        model = pickle.load(f)
        models_marathi.append(model)

gujarati_lexicon = pd.read_csv("./data/gujarati_data.csv")
models_gujarati = []
# model_files = ["./models/GBC_gujarati.pickle", "./models/LogReg_gujarati.pickle", "./models/SupportVec_gujarati.pickle",
#                "./models/RandForest_gujarati.pickle", "./models/SupportVec_gujarati.pickle", "./models/KNeighbors_gujarati.pickle"]
model_files = ["./models/LogReg_gujarati.pickle", "./models/SupportVec_gujarati.pickle",
               "./models/RandForest_gujarati.pickle", "./models/KNeighbors_gujarati.pickle"]
for file in model_files:
    with open(file, "rb") as f:
        model = pickle.load(f)
        models_gujarati.append(model)

hindi_lexicon = pd.read_csv("./data/hindi_data.csv")
models_hindi = []
model_files = ["./models/LogReg_hindi.pickle",
               "./models/SupportVec_hindi.pickle", "./models/RandForest_hindi.pickle"]
for file in model_files:
    with open(file, "rb") as f:
        model = pickle.load(f)
        models_hindi.append(model)

with open("./models/vectorizer_marathi.pickle", "rb") as f:
    vectorizer_marathi = pickle.load(f)
with open("./models/vectorizer_gujarati.pickle", "rb") as f:
    vectorizer_gujarati = pickle.load(f)
with open("./models/vectorizer_hindi.pickle", "rb") as f:
    vectorizer_hindi = pickle.load(f)


def punjabi_sentiment_score(text, lexicon):
    words = word_tokenize(text)
    positive_score = 0
    negative_score = 0
    for word in words:
        if word in lexicon['Word'].values:
            positive_score += lexicon.loc[lexicon['Word']
                                          == word, 'Positive Score'].values[0]
            negative_score += lexicon.loc[lexicon['Word']
                                          == word, 'Negative Score'].values[0]
            finalScore = positive_score - negative_score
    return positive_score - negative_score


def mr_gu_hi_sentiment(text, models, vectorizer):
    sentence = vectorizer.transform([text])
    predictions = []
    for model in models:
        prediction = model.predict(sentence)
        predictions.append(prediction)
    # Take the most common prediction
    sentiment = max(set(map(tuple, predictions)), key=predictions.count)
    return sentiment

import streamlit as st

def positive():
    st.markdown("<h2 style='text-align:center; color:green'>Sentence is Positive</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-size:80px; text-align:center'>😊</h2>", unsafe_allow_html=True)
    st.balloons()
    
def negative():
    st.markdown("<h3 style='text-align:center; color:red'>Sentence is Negative</h3>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-size:80px; text-align:center'>😔</h2>", unsafe_allow_html=True)

def neutral():
    st.markdown("<h3 style='text-align:center; color:blue'>Sentence is Neutral</h3>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-size:80px; text-align:center'>😐</h2>", unsafe_allow_html=True)
    st.snow()


def run_streamlit_app():
    st.set_page_config(page_title="Sentiment Analysis for Devenagari Language", page_icon="🧊", initial_sidebar_state="auto")

    st.markdown("<h1 style='text-align:center;'>Sentiment Analysis for Devenagari Language</h1>", unsafe_allow_html=True)
    text_input = st.text_input("Enter Text:")
    text_input = clear_text(text_input)
    if text_input:
        language = detect(text_input)
        if language == 'mr':
            st.markdown("<h2 style='text-align:center; font-size:15px'>The language is Marathi</h2>", unsafe_allow_html=True)
            sentiment = mr_gu_hi_sentiment(text_input, models_marathi, vectorizer_marathi)
            sentiment = sentiment[0]
            if sentiment == 0:
                neutral()
            elif sentiment == 1:
                positive()
            else:
                negative()

        elif language == 'hi':
            st.markdown("<h2 style='text-align:center; font-size:15px'>The language is Hindi</h2>", unsafe_allow_html=True)
            sentiment = mr_gu_hi_sentiment(text_input, models_hindi, vectorizer_hindi)
            sentiment = sentiment[0]
            if sentiment == 0:
                neutral()
            elif sentiment == 1:
                positive()
            else:
                negative()  

        elif language == 'pa':
            st.markdown("<h2 style='text-align:center; font-size:15px'>The language is Punjabi</h2>", unsafe_allow_html=True)
            score = punjabi_sentiment_score(text_input, punjabi_lexicon)
            if score > 0.15:
                positive()
            elif score == 0:
                neutral()
            else:
                negative()

        elif language == 'gu':
            st.markdown("<h2 style='text-align:center; font-size:15px'>The language is Gujarati</h2>", unsafe_allow_html=True)
            sentiment = mr_gu_hi_sentiment(text_input, models_gujarati, vectorizer_gujarati)
            if sentiment == 0:
                negative()
            else:
                positive()
        else:
            st.warning("The language is not Detected.")

run_streamlit_app()