import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Ensure stopwords are downloaded
nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load('model/spam_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')


stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def predict_spam(text):
    cleaned_text = preprocess(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    return "Spam" if prediction == 1 else "Ham"

# Example usage:
if __name__ == "__main__":
    sample = "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim now."
    print("Prediction:", predict_spam(sample))







