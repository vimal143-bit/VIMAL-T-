# Decoding Emotions Through Sentiment Analysis of Social Media Conversations
# Author: [Your Name]
# Dataset: Replace with actual dataset path or use a Kaggle dataset with 'text' and 'emotion' columns

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv('emotion_dataset.csv')  # Ensure the file exists in your path
print("Sample data:")
print(df.head())

# Basic info
print("Dataset info:")
print(df.info())

# Check for nulls
print("Missing values:")
print(df.isnull().sum())

# Drop missing values if any
df.dropna(inplace=True)

# Data distribution
sns.countplot(data=df, x='emotion')
plt.title("Distribution of Emotions")
plt.xticks(rotation=45)
plt.show()

# Text Cleaning Function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"http\S+|@\w+|#[A-Za-z0-9_]+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Apply cleaning
df['clean_text'] = df['text'].apply(clean_text)

# Print cleaned data
print("Cleaned text sample:")
print(df[['text', 'clean_text']].head())

# Text Vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text']).toarray()
y = df['emotion']

# Encode target if needed
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance using coefficients
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_
for i, emotion in enumerate(le.classes_):
    print(f"Top words for {emotion} emotion:")
    top_features = np.argsort(coefficients[i])[-10:]
    print([feature_names[j] for j in top_features])

# Save model and vectorizer (optional)
import joblib
joblib.dump(model, 'emotion_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Load and test (optional)
# model = joblib.load('emotion_model.pkl')
# vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Test on new sentence
def predict_emotion(text):
    text_clean = clean_text(text)
    vec = vectorizer.transform([text_clean]).toarray()
    pred = model.predict(vec)
    return le.inverse_transform(pred)[0]

# Example
sample = "I am feeling really happy today!"
print(f"Input: {sample}")
print("Predicted Emotion:", predict_emotion(sample))
# Additional code or comments...
# Additional code or comments...
# Additional code or comments...
# Additional code or comments...
# Additional code or comments...
# Additional code or comments...
# Additional code or comments...
# Additional code or comments...
# Additional code or comments...
# Additional code or comments...
# Additional code or comments...
# Additional code or comments...
# Additional code or comments...
# Additional code or comments...
# Additional code or comments...
# Additional code or comments...
# Additional code or comments...
# Additional code or comments...
# Additional code or comments...
# Additional code or comments...
# Additional code or comments...
# Additional code or comments...
# Additional code or comments...
# Additional code or comments...
# Additional code or comments...
# Additional code or comments...
# Additional code or comments...
# Additional code or comments...