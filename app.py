from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

print("Starting Flask app...")

# Initialize Flask app
app = Flask(__name__)

try:
    print("Loading model and vectorizer...")
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    print("Model and vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading model/vectorizer: {e}")
    model, vectorizer = None, None

# Initialize text preprocessing tools
nltk.download("stopwords")
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    """Text preprocessing: lowercase, remove special chars, stopwords, and lemmatize."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

print("Flask app initialized successfully.")

# Home route
@app.route("/")
def home():
    print("Home page accessed.")
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    print("Received prediction request.")

    review = request.form.get("review", "").strip()
    if not review:
        print("No review text provided.")
        return render_template("result.html", review="No input provided", sentiment="Error")

    cleaned_review = preprocess_text(review)
    print(f"Cleaned review: {cleaned_review}")

    try:
        review_vector = vectorizer.transform([cleaned_review])
        prediction = model.predict(review_vector)[0]
        sentiment_label = "Positive" if prediction == "1" else "Negative"
        print(f"Prediction result: {sentiment_label}")

        return render_template("result.html", review=review, sentiment=sentiment_label)
    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template("result.html", review=review, sentiment="Error")

# Run the Flask app
if __name__ == "__main__":
    print("Running Flask server...")
    app.run(debug=True)
