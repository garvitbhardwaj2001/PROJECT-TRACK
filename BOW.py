#BOW FOR TASK-1
import pandas as pd
import numpy as np

#Importing the dataset
dataset = pd.read_csv('imdb_master.csv', encoding = 'Latin-1')
X = dataset.iloc[0:25000,2].values
y = dataset.iloc[0:25000,3].values

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 25000):
    review = re.sub('[^a-zA-Z]', ' ', X[i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)