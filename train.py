# Load necessary libraries
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from src.preprocessing import preprocess_text
from src.feature_engineering import create_tfidf, transform_text, save_vectorizer, load_vectorizer
from src.model_training import train_model, evaluate_model, save_model, load_model

# Dataset path and model folder
DATA_PATH = "data/medicine_reviews.csv"
MODEL_FOLDER = "models"

os.makedirs(MODEL_FOLDER, exist_ok=True)

print("Loading data...")

df = pd.read_csv(DATA_PATH)
print(df.head())
# Define the text (independent) and target (dependent) columns
TEXT_COLUMN = "review"
TARGET_COLUMN = "label"

# Preprocessing the data. Dropping rows with missing values in the text and target columns
# df = df[[TEXT_COLUMN, TARGET_COLUMN]].dropna(inplace=True)

# Test with one row
# text = df[TEXT_COLUMN][0]
# result = preprocess_text(text)
# print(f"The Actual text: {text} and the preprocessed text is: {result}")

df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(preprocess_text) 

df[TARGET_COLUMN] = df[TARGET_COLUMN].map({'positive':1,'negative':0}) 

X_train, X_test, y_train, y_test = train_test_split( df[TEXT_COLUMN], df[TARGET_COLUMN],test_size=0.2, random_state=42)

vectorizer, X_train_tfidf = create_tfidf(X_train)

X_test_tfidf = transform_text(vectorizer, X_test)

print(f"Training the model")

model = train_model(X_train_tfidf, y_train) 

evaluate_model(model, X_test_tfidf, y_test)

print(f"Saving the Model & Vectorizer...")

save_vectorizer(vectorizer, "models/tfidf_vectorizer.pkl")

save_model(model, "models/random_forest.pkl")

print(f"Training complete. Model and Vectorizer saved in the models folder.")

