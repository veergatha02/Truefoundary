from fastapi import FastAPI
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
import pickle
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
#HELLOOO DERE
# load the vectorizer
loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))
# load the model
loaded_model = pickle.load(open('classification.model', 'rb'))
    
app = FastAPI()


@app.get('/')
def get_root():
    return {'message': 'This is the sentiment analysis app'}


@app.get('/sentiment_analysis/')
async def query_sentiment_analysis(text: str):
    return analyze_sentiment(text)


def analyze_sentiment(text):
    """Get and process result"""
    #test = countvector.transform([text])
    result = loaded_model.predict(loaded_vectorizer.transform([text]))

    sent = ''
    if (result == 'negative'):
        sent = 'negative'
    
    else:
        sent = 'positive'

    #prob = result[0]['score']

    # Format and return results
    return {'sentiment': sent}