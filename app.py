import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download stopwords
nltk.download('stopwords')

# LOAD DATA

@st.cache_resource
def load_model():
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")

    fake["label"] = 0
    true["label"] = 1

    data = pd.concat([fake, true])
    data = data.sample(frac=1, random_state=42)

    data["content"] = data["text"]

    stop_words = set(stopwords.words("english"))

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        words = text.split()
        words = [w for w in words if w not in stop_words]
        return " ".join(words)

    data["content"] = data["content"].apply(clean_text)

    X = data["content"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    return model, vectorizer, clean_text

model, vectorizer, clean_text = load_model()

# CATEGORY FUNCTION

def get_category(news):
    news = news.lower()

    if "alien" in news or "zombie" in news:
        return "Fake/Unrealistic"
    elif any(word in news for word in ["government", "policy", "minister", "modi"]):
        return "Politics"
    elif any(word in news for word in ["economy", "market", "bank"]):
        return "Economy"
    elif any(word in news for word in ["health", "hospital"]):
        return "Health"
    elif any(word in news for word in ["technology", "ai"]):
        return "Technology"
    else:
        return "General"

# UI DESIGN

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection System")
st.markdown("### 🔍 Check whether a news is REAL or FAKE using AI")

# Input box
news_input = st.text_area(" Enter News Here:", height=150)

if st.button("Analyze News"):
    if news_input.strip() == "":
        st.warning("Please enter some news text!")
    else:
        cleaned = clean_text(news_input)
        vec = vectorizer.transform([cleaned])

        prob = model.predict_proba(vec)[0]
        confidence = max(prob) * 100

        if prob[1] > prob[0]:
            label = "REAL NEWS"
            st.success(label)
        else:
            label = "FAKE NEWS"
            st.error(label)

        category = get_category(news_input)

        st.info(f" Confidence: {confidence:.2f}%")
        st.info(f" Category: {category}")

# Footer
st.markdown("---")
st.markdown("Developed using Machine Learning | Final Year Project")