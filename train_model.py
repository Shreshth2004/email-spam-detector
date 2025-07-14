import pandas as pd
import re
import nltk
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Download NLTK stopwords
nltk.download('stopwords')


# Load the dataset
df = pd.read_csv('Data/spam.csv', encoding='latin1')[['v1', 'v2']]
df.columns = ['label', 'message']


# Encode labels: spam = 1, ham = 0
df['label'] = df['label'].map({'ham': 0, 'spam': 1})


# Preprocessing function
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




# Apply preprocessing
df['message'] = df['message'].apply(preprocess)



# Split data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)



# Vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)


# Evaluate
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")


# Save model and vectorizer
joblib.dump(model, 'model/spam_model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')






