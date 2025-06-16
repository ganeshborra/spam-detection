import nltk

# Download necessary NLTK resources
nltk.download("punkt")


import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load model and vectorizer
model = joblib.load("sms_spam_detector.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    tokens = word_tokenize(text)  # Tokenization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]  # Lemmatization & stopword removal
    return ' '.join(tokens)

# Streamlit UI
st.title("📩 SMS Spam Detector")
st.write("Enter a message below to check if it's spam or not.")

# Input box
user_input = st.text_area("Enter SMS Message", "")

if st.button("Check Spam"):
    if user_input:
        processed_text = preprocess_text(user_input)
        transformed_text = vectorizer.transform([processed_text])
        prediction = model.predict(transformed_text)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        st.success(f"Prediction: **{result}**")
    else:
        st.warning("Please enter a message.")