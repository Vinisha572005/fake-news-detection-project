# ==========================================
# FINAL FAKE NEWS DETECTION (CLEAN VERSION)
# ==========================================

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download stopwords (only first time)
nltk.download('stopwords')

# ==========================================
# 1. LOAD DATA
# ==========================================
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# ==========================================
# 2. USE TEXT ONLY (BEST)
# ==========================================
data["content"] = data["text"]

# ==========================================
# 3. CLEAN TEXT
# ==========================================
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

data["content"] = data["content"].apply(clean_text)

# ==========================================
# 4. SPLIT
# ==========================================
X = data["content"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 5. TF-IDF
# ==========================================
vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==========================================
# 6. MODEL
# ==========================================
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ==========================================
# 7. CATEGORY FUNCTION (IMPROVED)
# ==========================================
def get_category(news):
    news = news.lower()

    if "alien" in news or "zombie" in news or "time travel" in news:
        return "Fake/Unrealistic"
    elif any(word in news for word in ["government", "policy", "minister", "election", "modi"]):
        return "Politics"
    elif any(word in news for word in ["economy", "market", "bank", "finance"]):
        return "Economy"
    elif any(word in news for word in ["health", "hospital", "disease", "medical"]):
        return "Health"
    elif any(word in news for word in ["technology", "ai", "software", "internet"]):
        return "Technology"
    else:
        return "General"

# ==========================================
# 8. PREDICTION FUNCTION (FIXED CONFIDENCE)
# ==========================================
def predict_news(news):
    cleaned = clean_text(news)
    vec = vectorizer.transform([cleaned])

    prob = model.predict_proba(vec)[0]

    confidence = max(prob) * 100

    if prob[1] > prob[0]:
        label = "REAL NEWS ✅"
    else:
        label = "FAKE NEWS ❌"

    category = get_category(news)

    return label, confidence, category

# ==========================================
# 9. INPUT LOOP
# ==========================================
while True:
    news = input("\n📰 Enter News (type 'exit' to quit): ")

    if news.lower() == "exit":
        print("👋 Exiting...")
        break

    label, conf, cat = predict_news(news)

    print("👉 Prediction :", label)
    print("📊 Confidence :", round(conf, 2), "%")
    print("🏷️ Category   :", cat)