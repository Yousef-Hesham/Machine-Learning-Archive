#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('capture', '', '#@title Setup Environment\n# Install the latest Tensorflow version.\n!pip3 install tensorflow_text')


# # Importing important libraries

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf


# In[ ]:


import tensorflow as tf

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


# In[ ]:


from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense


# # Reading dataset

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


df=pd.read_csv('/content/drive/MyDrive/help/workong remote/mlarchive/data/drugsComTrain_raw.csv')


# In[ ]:


# here we are printing first five lines of our train dataset
df.head()


# # Data cleaning

# In[ ]:


#filling nan values with space(' ')
df.fillna(' ',inplace=True)


# In[ ]:


df.isnull().sum()


# In[ ]:


df.head()


# In[ ]:


df['labels'] = df['rating'].apply(lambda rating : +1 if rating > 5 else -1)
df.head()


# In[ ]:


x=df['review']
y=df['labels']


# In[ ]:


x.head()


# In[ ]:


x[1]


# In[ ]:


from sklearn.model_selection import train_test_split
import numpy as np
#X_train, X_test, y_train, y_test = train_test_split(np.array(corpus),np.array(y), stratify=y)
X_train, X_test, y_train, y_test = train_test_split(x,y, stratify=y)


# #Now lets import BERT model and get embeding vectors for few sample statements

# In[ ]:


import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text


# In[ ]:


bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")


# In[ ]:


def get_sentence_embeding(sentences):
    preprocessed_text = bert_preprocess(sentences)
    return bert_encoder(preprocessed_text)['pooled_output']


# #Build Model

# In[ ]:


# Bert layers
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)

# Neural network layers
l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)

# Use inputs and outputs to construct a final model
model = tf.keras.Model(inputs=[text_input], outputs = [l])


# In[ ]:


model.summary()


# In[ ]:


len(X_train)


# In[ ]:


METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall')
]

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=METRICS)


# In[ ]:


X_train


# In[ ]:


model.fit(X_train, y_train, epochs=5)


# In[ ]:


model.evaluate(X_test, y_test)

