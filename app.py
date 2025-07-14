import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# 🚨 Ensure stopwords are downloaded at runtime
@st.cache_resource
def get_stopwords():
    try:
        return set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        return set(stopwords.words('english'))

stop_words = get_stopwords()

# Load trained model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Preprocessing function
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.title("📧 Email Spam Detector")
st.write("Enter a message and I'll tell you if it's spam or ham.")

# Text input
user_input = st.text_area("Your message:", height=150)

# Button to trigger prediction
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        processed = preprocess(user_input)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]
        label = "🟢 Ham (Not Spam)" if prediction == 0 else "🔴 Spam"
        st.markdown(f"## Prediction: {label}")
