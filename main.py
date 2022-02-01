import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

def pre_process(sentence):
    new_sentence = sentence[:len(sentence) - 3]
    return new_sentence.lower()


# Load data
X = []
Y = []
filenames = ['sentiment labelled sentences 2/amazon_cells_labelled.txt', 'sentiment labelled sentences 2/imdb_labelled.txt',
         'sentiment labelled sentences 2/yelp_labelled.txt']
for f in filenames:
    file = open(f, 'r')
    lines = file.readlines()
    for line in lines:
        sentence = pre_process(line)
        X.append(sentence)
        Y.append(line[-2])
    file.close()
#print(len(X))


# split data
x_train, x_test, y_train, y_test = train_test_split(X,Y, stratify=Y, test_size=0.25, random_state=0)
# Vectorize text data to numbers
vec = CountVectorizer(stop_words='english')
x_train = vec.fit_transform(x_train).toarray()
x_test = vec.transform(x_test).toarray()

# Create and Train Model
model = MultinomialNB()
model.fit(x_train, y_train)

# display results
y_pred = model.predict(x_test)
score = model.score(x_test, y_test)
print("Sentiment Analysis using Naive Bayes with Scikit-learn")
print("...predictions:\n", y_pred[0:15],"...")
print("...correct output:\n", y_test[0:15],"...")
print("...accuracy:", score)