# -*- coding: utf-8 -*-
"""Copy of fake.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1T16QKettcF_c_qKR7myNtCl_-NSj3AU_
"""

from google.colab import drive
drive.mount('/content/drive')

"""# Introduction


In this analysis, we will discuss how you can use NLP to determine whether the news is real or fake. Nowadays, fake news has become a common problem. Even respected media organizations are known to propagate fake news and are losing credibility. It can be difficult to trust news, because it can be difficult to know whether a news story is real or fake.

# Dataset
1.train.csv: A full training dataset with the following attributes                                         
2.id: unique id for a news article                                                                         
3.title: the title of a news article                                                                       
4.author: author of the news article                                                                       
5.text: the text of the article; could be incomplete                                                       
6.label: a label that marks the article as potentially unreliable. Where 0: reliable and 1: unreliable.

# Importing important libraries
"""

import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

# here we are importing nltk,stopwords and porterstemmer we are using stemming on the text
# we have and stopwords will help in removing the stopwords in the text

#re is regular expressions used for identifying only words in the text and ignoring anything else
import nltk
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
ps=PorterStemmer()

"""# Reading dataset"""

train_df=pd.read_csv('/content/drive/MyDrive/research/aireserchnlp/fakenews/try2/fakdata/train.csv')

# here we are printing first five lines of our train dataset
train_df.head()

"""# Data Pre-Processing

#preprocessing
"""

#filling nan values with space(' ')
train_df.fillna(' ',inplace=True)

#combining title and author,title and summary is formed
train_df['summary']=train_df['title']+' '+train_df['author']+' '+train_df['text']
train_df.head()

train_df['summary'][1]

train_df.isnull().sum()

train_df['summary']==' '

"""**Removel of stop words and Stemming the words**"""

x=train_df['summary']
y=train_df['label']

x.head()

x[1]

import nltk
nltk.download('stopwords')

# here we are creating corpus for the test dataset exactly the same as we created for the
# training dataset
corpus=[]
for i in range(0,len(train_df)):
    review=re.sub('[^a-zA-Z]',' ',x[i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)

corpus[1]

"""**Word Embedding — One hot encoding**

The machine cannot understand words and therefore it needs numerical values so as to make it easier for the machine to process the data. To apply any type of algorithm to the data, we need to convert the categorical data to numbers. To achieve this, one hot ending is one way as it converts categorical variables to binary vectors.
"""

#vocabulary size
voc_size=10000

# TensorFlow has an operation for one-hot encoding
one_hot_reps1=[one_hot(word,voc_size) for word in corpus]
one_hot_reps1[1]

"""**Word Embedding**"""

# here we are specifying a sentence length so that every sentence in the corpus will be of same length
sent_length=500
#making all the sentence as equall size vector
#two types of padding pre and post
embedded_docs1=pad_sequences(one_hot_reps1,padding='pre',maxlen=sent_length)
embedded_docs1

x=np.array(embedded_docs1)
y=np.array(y)

x

""",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

# LSTM

# Building Models
"""

#Creating model
from tensorflow.keras.layers import Dropout
import warnings
warnings.filterwarnings('ignore')
embedded_feature_vector=300
nn=Sequential([

    Embedding(voc_size,embedded_feature_vector,input_length=sent_length),
    Dropout(0.5),
    LSTM(199),
    Dropout(0.4),
    Dense(399,activation='relu'),
    Dense(43,activation='relu'),
    Dense(1,activation='sigmoid')])

nn.summary()

"""# Spiliting and Training"""

# here we are splitting the data for training and testing the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

nn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#nn.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=50,batch_size=64)
# Train the model on the training data with validation split
nn.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=64)

y_pred=nn.predict(X_test)
y_pred

y_pred=(y_pred>0.5)
y_pred

y_pred=y_pred.reshape(-1,)
y_pred

y_test

#y_pred=np.argmax(y_pred)

#from sklearn.preprocessing import label_binarize
#y_pred=label_binarize(y_pred, classes=['False', 'True'])

y_pred= np.array(y_pred)
y_pred

y_test =np.array(y_test)
y_test

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
cm

from matplotlib import pyplot as plt
import seaborn as sn
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))