import logging
import pickle
import json

from fastapi import FastAPI, File
from pydantic import BaseModel

import numpy as np
import pandas as pd
import sklearn


app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add variables
    """
    # Load model and preprocess tools
    with open("/code/app/logistic_classifier.pickle", "rb") as f:
        classifier = pickle.load(f)
    with open("/code/app/binarizer.pickle", "rb") as f:
        binarizer = pickle.load(f)
    with open("/code/app/tfidf.pickle", "rb") as f:
        vectorizer = pickle.load(f)
    with open("/code/app/token2id.pickle", "rb") as f:
        tokens_set = pickle.load(f)


    # add model and preprocess tools too app state
    app.package = {
        "classifier": classifier,
        "binarizer": binarizer,
        "vectorizer": vectorizer,
        "tokens_set": tokens_set
    }


class Question(BaseModel):
    title: str
    body: str | None = None


@app.post("/api/v1/predict")
def do_predict(question : Question):
    text = question.title + " " + question.body if  question.body else question.title
    text = text.strip().lower()
    doc = [word for word in text.split(" ") if word.isalnum()]
    tokens = [token for token in doc if token in app.package["tokens_set"]]
    doc_vectorized = app.package["vectorizer"].transform(pd.Series([" ".join(tokens)]))
    y_pred_proba = app.package["classifier"].predict_proba(doc_vectorized)
    y_best_preds = []
    for pred in y_pred_proba:
        best_preds_indexes = np.argsort(pred)[-3:]
        results = np.zeros((pred.shape[0],))
        results[best_preds_indexes] = 1
        y_best_preds.append(results)
    y_best_preds = np.array(y_best_preds)
    tags = app.package["binarizer"].inverse_transform(y_best_preds)[0]

    return {"text": text, "tags": tags}

"""
@app.route('/')
def home():
    return 'Flask with docker!'

@app.route("/predict_tags", methods=['POST'])
def predict_tags():
    data = request.json
    title = data["title"] if "title" in data.keys() else ""
    body = data["body"] if "body" in data.keys() else ""
    text = title + " " + body
    logger.debug(f"Request text : {text}")
    text = text.strip().lower()
    doc = [word for word in text.split(" ") if word.isalnum()]
    tokens = [token for token in doc if token in tokens_set]
    logger.debug(f"Request tokens : {tokens}")
    logger.debug(pd.Series([" ".join(tokens)]))
    doc_vectorized = vectorizer.transform(pd.Series([" ".join(tokens)]))
    logger.debug(f"Doc vectorized : {doc_vectorized[0].todense()}")
    logger.debug(type(doc_vectorized))
    y_pred_proba = classifier.predict_proba(doc_vectorized)
    logger.debug(y_pred_proba)
    y_best_preds = []
    for pred in y_pred_proba:
        best_preds_indexes = np.argsort(pred)[-3:]
        results = np.zeros((pred.shape[0],))
        results[best_preds_indexes] = 1
        y_best_preds.append(results)
    y_best_preds = np.array(y_best_preds)
    tags = binarizer.inverse_transform(y_best_preds)[0]
    return jsonify({"text": text, "tags": tags})

if __name__ == '__main__':
     classifier = pickle.load("logistic_classifier.pickle")
     logger.info("!!!!!!!!!!!!!!!!!!!!!")
     logger.info(type(classifier))

     app.run(port=8080)
"""