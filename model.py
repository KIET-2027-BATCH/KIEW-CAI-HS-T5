import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ✅ Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# ✅ Initialize text processing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ✅ Text Preprocessing Function
def preprocess_text(text):
    if isinstance(text, str):  
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
        words = text.split()  
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  
        
        # Ensure empty text is replaced with a default value
        return ' '.join(words) if words else "emptytext"
    
    return "emptytext"

# ✅ Load Dataset
file_path = r"C:\Users\R HARIKA\Downloads\PRODUCT REVIEWS\Amazon-Product-Reviews - Amazon Product Review (1).csv"
if not os.path.exists(file_path):
    print("❌ Error: Dataset file not found. Check file path.")
    exit()

df = pd.read_csv(file_path, encoding='utf-8')

# ✅ Check dataset contents
if df.empty:
    print("❌ Error: The dataset is empty. Ensure correct file is used.")
    exit()

# ✅ Drop missing values
df.dropna(subset=['review_body', 'sentiment'], inplace=True)

# ✅ Apply Preprocessing
df['cleaned_text'] = df['review_body'].astype(str).apply(preprocess_text)

# ✅ Debugging: Check dataset after preprocessing
print("\n📌 First 5 cleaned reviews:")
print(df[['review_body', 'cleaned_text']].head())

# ✅ Ensure dataset is not empty after preprocessing
if df['cleaned_text'].str.strip().eq('emptytext').all():
    print("❌ Error: All reviews are empty after preprocessing. Adjust preprocessing function.")
    exit()

# ✅ Handle dataset imbalance
df['sentiment'] = df['sentiment'].astype(str)  
min_count = df['sentiment'].value_counts().min()  

if min_count < 5:
    print("❌ Error: Some sentiment classes have very few samples. Consider removing them or collecting more data.")
    exit()

df_balanced = df.groupby('sentiment', group_keys=False).apply(lambda x: x.sample(n=min_count, random_state=42))

# ✅ Ensure balanced dataset is not empty
if df_balanced.empty:
    print("❌ Error: Dataset is empty after balancing. Check sentiment distribution.")
    exit()

# ✅ Vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

try:
    X = vectorizer.fit_transform(df_balanced['cleaned_text']).toarray()
    y = df_balanced['sentiment'].values
except ValueError as e:
    print(f"❌ TF-IDF Error: {e}")
    exit()

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# ✅ Evaluate Model
y_pred = model.predict(X_test)
print("\n🔹 Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))

# ✅ User Input Prediction
def predict_sentiment():
    user_input = input("\nEnter a product review: ")
    cleaned_input = preprocess_text(user_input)  
    input_vector = vectorizer.transform([cleaned_input]).toarray()  
    prediction = model.predict(input_vector)[0]  
    
    # ✅ Convert 1 → Positive, 0 → Negative in output
    sentiment_label = "Positive" if prediction == "1" else "Negative"
    print(f"\nPredicted Sentiment: {sentiment_label} ({prediction})")

predict_sentiment()
import joblib

# Save Model
joblib.dump(model, "sentiment_model.pkl")

# Save Vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")