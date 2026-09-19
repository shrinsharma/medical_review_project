from sklearn.feature_extraction.text import TfidfVectorizer

import joblib

def create_tfidf(X_train): 
	vectorizer = TfidfVectorizer() 
	X_train_tfidf = vectorizer.fit_transform(X_train) 
	return vectorizer,X_train_tfidf

def transform_text(vectorizer,X_test): 
	return vectorizer.transform(X_test)

def save_vectorizer(vectorizer, path):
	joblib.dump(vectorizer, path)
	return "Successfully saved the Vectorizer to the path: {}".format(path)

def load_vectorizer(path):
	return joblib.load(path)

