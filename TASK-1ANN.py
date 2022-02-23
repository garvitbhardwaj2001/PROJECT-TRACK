import pandas as pd
import numpy as np
import pickle

#Importing the dataset
dataset = pd.read_csv('imdb_master.csv', encoding = 'Latin-1')
X = dataset.iloc[0:25000,2]
y = dataset.iloc[0:25000,3].values

#Importing TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
filename = 'TF-IDF.sav'
pickle.dump(vectorizer, open(filename, 'wb'))
review = [review]
#vectorizer = pickle.load(open('TF-IDF.sav', 'rb'))
review = vectorizer.transform(review)

for i in range(25000):
    if y[i]=='pos':
        y[i]=1
    elif y[i]=='neg':
        y[i]=0
        
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
input_dim = X_train.shape[1]

# Importing the Keras libraries and packages
import os
os.environ['KERAS_BACKEND'] = 'theano'
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialisation
classifier = Sequential()
classifier.add(Dense(10, input_dim=input_dim, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))

# Compiling
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN
classifier.fit(X_train, y_train, batch_size = 1000, epochs = 30)

y_pred = classifier.predict(X_test)
y_pred = y_pred.round()
y_test = y_test.tolist()

from sklearn.metrics import accuracy_score
report = accuracy_score(y_test,y_pred)
print(report)