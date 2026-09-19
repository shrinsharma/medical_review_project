# Loading necessary libraries   
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download('punkt_tab')

# Initialize the lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#Preprocessing function to clean and preprocess the text data
def preprocess_text(text):
  text = text.lower()
  text = re.sub(r'[^a-z\s]','',text)
  tokens = word_tokenize(text)
  filtered_tokens = []
  for word in tokens:
    if word not in stop_words:
      filtered_tokens.append(word)
      lemmatized_tokens = []
      for word in filtered_tokens:
        lemmatized_tokens.append(lemmatizer.lemmatize(word))
    return ' '.join(lemmatized_tokens)