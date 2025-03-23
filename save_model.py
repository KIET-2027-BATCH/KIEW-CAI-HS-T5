import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ✅ Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# ✅ Initialize text processing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ✅ Text Preprocessing Function
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return ' '.join(words) if words else "emptytext"
    return "emptytext"

# ✅ Load Dataset
file_path = r"C:\Users\R HARIKA\Downloads\PRODUCT REVIEWS\Amazon-Product-Reviews - Amazon Product Review (1).csv"
df = pd.read_csv(file_path, encoding='utf-8')

# ✅ Drop missing values
df.dropna(subset=['review_body', 'sentiment'], inplace=True)

# ✅ Apply Preprocessing
df['cleaned_text'] = df['review_body'].astype(str).apply(preprocess_text)

# ✅ Handle dataset imbalance
df['sentiment'] = df['sentiment'].astype(str)
min_count = df['sentiment'].value_counts().min()
df_balanced = df.groupby('sentiment', group_keys=False).apply(lambda x: x.sample(n=min_count, random_state=42))

# ✅ Vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df_balanced['cleaned_text']).toarray()
y = df_balanced['sentiment'].values

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# ✅ Save Model and Vectorizer as Pickle Files
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")
