# required libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SimpleRNN

# Load data
data = pd.read_csv('btc_15m.csv')
data.drop(columns='timestamp',inplace=True)
data.head()

# target and more features calculations
data['target'] = (np.sign(data['close']-data['open'])+1)/2
data['f1'] = data['high']-data['open']
data['f2'] = data['close']-data['low']
print(data.head())

print(data.info())

# feature and target separation
X = data.drop(columns=['open','close','target'])
y = data['target']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# data normalization
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build LSTM model
model1 = Sequential()
model1.add(LSTM(64, activation='relu',input_shape=(X_train.shape[1],1)))
model1.add(Dropout(0.2))
model1.add(Dense(units=8,activation='relu'))
model1.add(Dense(units=1,activation='tanh'))

# Compile model
model1.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


# Train model
history = model1.fit(X_train, y_train, epochs=30, batch_size=1024, verbose=None)

# Evaluate model
loss1, accuracy1 = model1.evaluate(X_test,y_test)
print(f'model_1 loss value is: {loss1} \n')
print(f'model_1 accuracy is: {accuracy1}')

# Build SimpleRNN model
model2 = Sequential()
model2.add(SimpleRNN(units=64, activation='relu', input_shape=(X_train.shape[1], 1), return_sequences=True))
model2.add(Dropout(0.2))
model2.add(SimpleRNN(units=32, activation='relu', return_sequences=True))
model2.add(Dropout(0.2))
model2.add(SimpleRNN(units=8, activation='relu'))
model2.add(Dense(units=1,activation='tanh'))

# Compile model
model2.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model2.fit(X_train, y_train, epochs=30, batch_size=1024, verbose=None)

# Evaluate model
loss2, accuracy2 = model2.evaluate(X_test,y_test)
print(f'model_2 loss value is: {loss2} \n')
print(f'model_2 accuracy is: {accuracy2}')
