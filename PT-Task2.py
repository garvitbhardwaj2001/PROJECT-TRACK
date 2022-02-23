import pandas as pd
import numpy as np

dataset = pd.read_csv('Tweets.txt' , sep = '\t')
X = dataset.iloc[:,1]
y = dataset.iloc[:,2:].values
tweet = str(input('How are you feeling today?'))

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
tweet = [tweet]
tweet = vectorizer.transform(tweet)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 0)

#print(X_train.shape)

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
y_pred = []
pred = []
for i in range(0,11):
    classifier = LinearSVC()
    classifier.fit(X_train,y_train[:,i])
    y_pred.append(classifier.predict(X_test))
    pred.append(classifier.predict(tweet))
    
y_pred = np.array(y_pred)
y_pred = y_pred.T
pred = np.array(pred)
pred = pred.T

list1 = list(dataset.columns.values)

for i in range(11):
    print(list1[i+2],pred[0][i])

#from sklearn.metrics import classification_report
#report = classification_report(y_test,y_pred)
#print(report)