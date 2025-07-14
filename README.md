# 📧 Email Spam Detector

A simple yet powerful machine learning web app to detect spam messages using NLP. It classifies messages as **Spam** or **Ham (Not Spam)** using a trained Naive Bayes classifier. Built with **Python**, **NLTK**, **scikit-learn**, and deployed with **Streamlit**.

---

## 🚀 Live Demo

👉 [Click here to try the app](https://email-spam-detector-sj.streamlit.app/)  


---

## 🧠 How It Works

1. Takes email/message text input from the user
2. Preprocesses the text (lowercasing, stopword removal, stemming, etc.)
3. Converts text into numerical features using **TF-IDF Vectorization**
4. Predicts using a trained **Multinomial Naive Bayes model**
5. Outputs whether the message is **Spam** or **Ham**

---

## 📁 Project Structure
SPAM_MAIL_DETECTOR/
├── app.py # Streamlit app code
├── train_model.py # Model training script
├── spam_detector.py # Prediction logic (optional utility script)
├── model/
│ ├── spam_model.pkl # Trained Naive Bayes model
│ └── vectorizer.pkl # Fitted TF-IDF vectorizer
├── Data/
│ └── spam.csv # Dataset of labeled SMS messages
├── requirements.txt # Python dependencies
├── .gitignore # Files/folders to ignore in Git
└── README.md # Project documentation


---

## 📦 Installation

### 🖥️ Run Locally

1. Clone the repository:
```bash
git clone https://github.com/Shreshth2004/email-spam-detector.git
cd email-spam-detector

python -m venv venv
venv\Scripts\activate  # On Windows

pip install -r requirements.txt

streamlit run app.py



