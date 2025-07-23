# train_models.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle
# Create the model_files directory if it doesn't exist
import os
os.makedirs("model_files", exist_ok=True)


# Load dataset
df = pd.read_csv("fake_job_postings-2.csv")

# Drop rows with missing required fields
df.dropna(subset=["title", "location", "description", "benefits"], inplace=True)

# Combine selected features
df["text"] = (
    df["title"].fillna("") + " "
    + df["location"].fillna("") + " "
    + df["description"].fillna("") + " "
    + df["benefits"].fillna("")
)

# Labels
X = df["text"]
y = df["fraudulent"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vect = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "randomforest": RandomForestClassifier(),
    "naivebayes": MultinomialNB(),
    "svm": LinearSVC(),
    "decisiontree": DecisionTreeClassifier()
}

accuracies = {}

# Train and save models
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    accuracies[name] = acc
    with open(f"model_files/{name}.pkl", "wb") as f:
        pickle.dump(model, f)

# Save vectorizer
with open("model_files/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save accuracies
with open("model_files/accuracies.pkl", "wb") as f:
    pickle.dump(accuracies, f)
