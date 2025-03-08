import streamlit as st
import pickle
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import shutil

# Define the NLTK data path
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")

# 🔴 Fix: Remove existing data directory to prevent conflicts
if os.path.exists(nltk_data_path):
    shutil.rmtree(nltk_data_path)  # Deletes the existing directory to prevent errors

# Recreate the directory
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Ensure required NLTK resources are downloaded
for resource in ["punkt", "stopwords"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_path)

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

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
