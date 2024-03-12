import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

# Data Collection and Preprocessing
# Assume you have a dataset 'data.csv' containing text data about India's geopolitics

# Load data
data = pd.read_csv('data.csv')

# Preprocess text data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(filtered_tokens)

data['processed_text'] = data['text'].apply(preprocess_text)

# Train a TF-IDF Vectorizer and a Classifier
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data['processed_text'])
y = data['label']

classifier = MultinomialNB()
classifier.fit(X, y)

# User Interface
def predict_geopolitics(text):
    processed_text = preprocess_text(text)
    X_test = tfidf_vectorizer.transform([processed_text])
    prediction = classifier.predict(X_test)
    return prediction[0]

# Example usage
user_input = input("Enter text: ")
prediction = predict_geopolitics(user_input)
print("Predicted label:", prediction)