import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()
tfidf = pickle.load(open('tfidfvectorizer.pkl', 'rb'))
model = pickle.load(open('SMSSpamDetectionClassifier.pkl', 'rb'))
st.title("Email/SMS Spam Classifier")


def transform_text(text):
    text = text.lower()  # lowering
    words = nltk.tokenize.word_tokenize(text)  # tokenization
    y = []
    for word in words:
        if word.isalnum():  # removing special characters
            y.append(word)
    text = y[:]
    y.clear()
    for word in text:  # removing stowords and punctuation
        if word not in stopwords.words('english') and word not in string.punctuation:
            y.append(ps.stem(word))  # stemming

    return " ".join(y)

input_sms = st.text_area('Enter the message')
if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")