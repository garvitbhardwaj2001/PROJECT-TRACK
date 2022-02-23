#For text preprocessing
import re
import spacy
#For handing csv and txt files
import pandas as pd  
from time import time
# For word frequency
from collections import defaultdict  
import numpy as np

# Setting up the loggings to monitor gensim  
import logging 
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

df = pd.read_csv('Tweets.txt',sep='\t')
df = df.dropna().reset_index(drop=True)
df.isnull().sum()
X = df.iloc[:,1].values
y = df.iloc[:,2:].values

nlp = spacy.load('en', disable=['ner', 'parser']) # disabling Named Entity Recognition for speed

def cleaner(doc):
    txt = [token.lemma_ for token in doc if not token.is_stop]
    if len(txt) > 2:
        return ' '.join(txt)

brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in X)
t = time()

txt = [cleaner(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1)]

df_clean = pd.DataFrame({'clean': txt})
df_clean = df_clean.dropna().drop_duplicates()
df_clean.shape

from gensim.models.phrases import Phrases
sent = [row.split() for row in df_clean['clean']]
phrases = Phrases(sent, min_count=30, progress_per=10000)
sentences = phrases[sent]

word_freq = defaultdict(int)
for sent in sentences:
    for i in sent:
        word_freq[i] += 1
len(word_freq)

from gensim.models import Word2Vec
w2v_model = Word2Vec(min_count=20,
                     window=2,
                     size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=11)
w2v_model.build_vocab(sentences, progress_per=10000)
t = time()

w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
w2v_model.init_sims(replace=True)

#save model
filename = 'simpsons_w2v.txt'
w2v_model.wv.save_word2vec_format(filename,binary=False)

import os
index = {}
f = open(os.path.join('',filename),encoding = 'utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    index[word] = coefs
f.close()

import os
os.environ['KERAS_BACKEND'] = 'theano'
#Importing keras for creation of neural network
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding

t = Tokenizer()
t.fit_on_texts(X)
vocab_size = len(t.word_index) + 1
encoded_docs = t.texts_to_sequences(X)
max_length = 30
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

matrix = np.zeros((vocab_size,300))

for word, i in word_freq.items():
    if i>vocab_size:
        continue
    vector = index.get(word)
    if vector is not None:
        matrix[i] = vector

from sklearn.model_selection import train_test_split
X_train,  X_test,y_train, y_test = train_test_split(padded_docs,y,test_size = 0.1, random_state=0)
y_pred = np.zeros((684,11))
for i in range(0,11):
    model = Sequential()
    e = Embedding(vocab_size, 300, weights=[matrix], input_length=30, trainable=False)
    model.add(e)
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
# compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.fit(X_train, y_train[:,i], epochs=20, verbose=0)
    y_pred[:,i] = np.array(model.predict(X_test).T)

print(len(w2v_model.wv.vocab))

y_pred = y_pred>=0.5
from sklearn.metrics import classification_report, confusion_matrix
report = classification_report(y_test,y_pred)
print(report)


        