import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('train.tsv/train1.csv')

vocab_size = 50000
temp=list(set(val for val in dataset['Phrase'].values))
print(len(temp))


tokenizer = Tokenizer(num_words=vocab_size,lower=True, split=" ",oov_token="<oov>")
tokenizer.fit_on_texts(dataset['Phrase'].values)
X = tokenizer.texts_to_sequences(dataset['Phrase'].values)
X = pad_sequences(X,maxlen=60,padding='post',truncating='post')
embed_dim=64

model = Sequential()
model.add(Embedding(vocab_size, embed_dim ,input_length = X.shape[1]))
model.add(LSTM(32, dropout=0.2, input_shape=(X.shape[1],64),recurrent_dropout=0.2,return_sequences=True))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(units=16,input_shape=(60,32)))
model.add(Dense(5,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())



Y = dataset['Sentiment'].values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 1)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
Y_train = Y_train.reshape(Y_train.shape[0],1)
Y_test=Y_test.reshape(Y_test.shape[0],1)

from keras.utils import to_categorical
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

batch_size = 32
model.fit(X_train, Y_train, epochs = 2, batch_size=batch_size,shuffle=True,verbose = 1,validation_split=0.2,validation_data=(X_test,Y_test))


score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

twt = ['This is not exceptable by the people.']
twt = tokenizer.texts_to_sequences(twt)
twt = pad_sequences(twt, maxlen=60, dtype='int32',padding='post',truncating='post', value=0)
print(twt)
sentiment = model.predict(twt,batch_size=1,verbose = 2)
print(sentiment)
print(np.argmax(sentiment))

fer1_json = model.to_json()
with open("fer1.json", "w") as json_file:
    json_file.write(fer1_json)
model.save_weights("fer1.h5")

# =============================================================================
# 0 - negative
# 1 - somewhat negative
# 2 - neutral
# 3 - somewhat positive
# 4 - positive
# =============================================================================

# =============================================================================
# model = Sequential()
# model.add(Embedding(X.shape[0], 128 ,input_length = X.shape[1]))
# model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
# model.add(SpatialDropout1D(0.2))
# model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
# model.add(SpatialDropout1D(0.2))
# model.add(Flatten())
# model.add(Dense(5,activation='softmax'))
# model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
# print(model.summary())
# =============================================================================

