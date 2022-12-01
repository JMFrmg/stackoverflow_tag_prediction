import logging
import pickle
import json

from flask import Flask, request, jsonify
import sklearn

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

with open("logistic_classifier.pickle", "rb") as f:
    classifier = pickle.load(f)
with open("binarizer.pickle", "rb") as f:
    binarizer = pickle.load(f)
with open("tfidf.pickle", "rb") as f:
    tfidf = pickle.load(f)
with open("token2id.pickle", "rb") as f:
    token2id = pickle.load(f)

logger.info("!!!!!!!!!!!!!!!!!!!!!")
logger.info(type(classifier))

app = Flask(__name__)

@app.route('/')
def home():
    return 'Flask with docker!'

@app.route("/predict_tags")
def predict_tags():
    text = request.json()
    logger.info(f"Request text : {text}")
    return jsonify({"text": text})

if __name__ == '__main__':
     classifier = pickle.load("logistic_classifier.pickle")
     logger.info("!!!!!!!!!!!!!!!!!!!!!")
     logger.info(type(classifier))

     app.run(port=8080)