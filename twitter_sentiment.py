import keras.losses
from keras.preprocessing.text import one_hot
from keras.layers import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
'''word_size=10000
sent=['He is a good boy',
      'he eats banana',
      'Rama killed Ravana',
      'He likes the village environment']
sequence=[one_hot(word,word_size)for word in sent]
sent_len=10
pad=pad_sequences(sequence,maxlen=sent_len)
print(pad)
dimen=100
mod=Sequential()
mod.add(Embedding(word_size,10,input_length=sent_len))
mod.add(keras.layers.Dense(units=1,activation='sigmoid'))
mod.add(keras.layers.LSTM(100))
mod.compile(optimizer='adam',loss=keras.losses.binary_crossentropy,metrics='accuracy')'''
from keras.layers import LSTM,Dense
import pandas as pd
data=pd.read_csv('Data.csv')
print(data.columns)
x=data['text']
import pandas as pd
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv('Data.csv')
lab=LabelEncoder()
data['sentiments']=lab.fit_transform(data['sentiment'])
y=data['sentiments']
print(y)
word_size=10000
onehot=[one_hot(word,word_size)for word in x]
sent_len=50
import numpy as np
sent=pad_sequences(onehot,maxlen=sent_len)
x=np.array(sent)
model=Sequential()
model.add(Embedding(word_size,100,input_length=sent_len))
model.add(LSTM(100))
model.add(Dense(1,activation='softmax'))
import keras.metrics,keras.optimizers
model.compile(loss=keras.losses.binary_crossentropy,metrics='accuracy',optimizer='adam')
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35)
model.fit(x_train,y_train,batch_size=10,epochs=12)
a=['this is not nice','how are you']
one=[one_hot(i,word_size) for i in a]
p=pad_sequences(one,maxlen=sent_len)
pr=np.array(p)
pre=model.predict(pr)
print(pre)