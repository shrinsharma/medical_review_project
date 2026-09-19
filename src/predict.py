import joblib

from src.preprocessing import preprocess_text
from src.feature_engineering import transform_text, load_vectorizer

def predict_review(review):
    # Load the pkl files
    model = joblib.load("models/random_forest.pkl")
    vectorizer = load_vectorizer("models/tfidf_vectorizer.pkl")

    #preprocessing the input review
    review = preprocess_text(review)

    # Vectorization of the input review
    review_vector = transform_text(vectorizer, [review])

    #Prediction
    prediction = model.predict(review_vector)[0]
    prob = model.predict_proba(review_vector).max()

    if prediction == 1:
        prediction = "Positive"
    else:
        prediction = "Negative"

    return prediction, prob

