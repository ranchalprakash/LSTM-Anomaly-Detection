#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 04:44:55 2018

@author: ranchal
"""
import matplotlib.pyplot as plt
import numpy as np
import time
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from numpy import sin, random


np.random.seed(1234)

# Global hyper-parameters
sequence_length = 20
random_data_dup = 10  # each sample randomly duplicated between 0 and 9 times, see dropin function
epochs =6
batch_size = 50
def dropin(X, y):

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    X_hat = []
    y_hat = []
    for i in range(0, len(X)):
        for j in range(0, np.random.random_integers(0, random_data_dup)):
            X_hat.append(X[i, :])
            y_hat.append(y[i])
    return np.asarray(X_hat), np.asarray(y_hat)


def signal_gen():
    t = np.arange(0.0, 10.0, 0.01)
    w1=10
    w2=30
    w3=50
    wave = 5*(sin(w1* t)+sin(w2* t)+sin(w3* t))
    print("wave", len(wave))
    n=round(len(wave)*.06) #Adding anomalies in 6% of training data
    anmly =[]
    for i in range(0,n):
        index=random.random_integers(low=0, high=len(wave)-1)
        value=np.random.uniform(low=-(wave.max()), high=(wave.max()))
        wave[index]=value
        anmly.append(t[index])
        
    anmly=np.sort(anmly)    
    return(wave,t,anmly)
def get_split_prep_data(train_start, train_end,
                          test_start, test_end):
    data,t,anmly =signal_gen()
    print("Length of Data", len(data))

    # train data
    print ("Creating train data...")

    result = []
    for index in range(train_start, train_end - sequence_length):
        result.append(data[index: index + sequence_length])
    result = np.array(result)  # shape (samples, sequence_length)
    print("Train data shape  : ", result.shape)

    train = result[train_start:train_end, :]
    
    np.random.shuffle(train)  # shuffles in-place
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_train, y_train = dropin(X_train, y_train)

    # test data
    print("Creating test data...")

    result = []
    for index in range(test_start, test_end - sequence_length):
        result.append(data[index: index + sequence_length])
    result = np.array(result)  # shape (samples, sequence_length)
    print("Test data shape  : ", result.shape)

    X_test = result[:, :-1]
    y_test = result[:, -1]

    print("Shape X_train", np.shape(X_train))
    print("Shape X_test", np.shape(X_test))

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


    return X_train, y_train, X_test, y_test,anmly,data,t


def build_model():
    model = Sequential()
    layers = {'input': 1, 'hidden1': 64, 'hidden2': 256, 'hidden3': 100, 'output': 1}

    model.add(LSTM(
            input_length=sequence_length-1,
            input_dim=layers['input'],
            output_dim=layers['hidden1'],
            return_sequences=True))
    model.add(Dropout(0.1))

    model.add(LSTM(
            layers['hidden2'],
            return_sequences=True))
    model.add(Dropout(0.1))

    model.add(LSTM(
            layers['hidden3'],
            return_sequences=False))
    model.add(Dropout(0.1))

    model.add(Dense(
            output_dim=layers['output']))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time.time() - start)
    return model

##############################################################################3
def run_network(model=None, data=None):
    global_start_time = time.time()

    if data is None:
        print('Loading data... ')
        # train on first 700 samples and test on next 300 samples (has anomaly)
        X_train, y_train, X_test, y_test,anmly,whole_data,t= get_split_prep_data(0, 700, 500, 1000)
    else:
        X_train, y_train, X_test, y_test = data

    print('\nData Loaded. Compiling...\n')

    if model is None:
        model = build_model()

    try:
        print("Training...")
        history=model.fit(
                X_train, y_train,
                batch_size=batch_size, nb_epoch=epochs, validation_split=0.1)
        print("Predicting...")
        predicted = model.predict(X_test)
        print("Reshaping predicted")
        predicted = np.reshape(predicted, (predicted.size,))
    except KeyboardInterrupt:
        print("prediction exception")
        print('Training duration (s) : ', time.time() - global_start_time)
        return model, y_test, 0

    try:
        fig=plt.figure(figsize=(10,13))
        plt.subplot(311)
        plt.title("Actual Test Signal w/Anomalies",fontsize=16)
        plt.plot(y_test[:len(y_test)], 'b')
        plt.subplot(312)
        plt.title("Predicted Signal",fontsize=16)
        plt.plot(predicted[:len(y_test)], 'g')
        plt.subplot(313)
        plt.title("Squared Error",fontsize=16)
        mse = ((y_test - predicted) ** 2)
        plt.plot(mse, 'r')
        plt.show()
        fig.savefig('foo.png', dpi=100)
    except Exception as e:
        print("plotting exception")
        print(str(e))
    print('Training duration (s) : ', time.time() - global_start_time)
    anmly_pred=[]
    mat=[]
    for index in range(sequence_length, len(whole_data)):
        mat.append(whole_data[index - sequence_length:index ])
    mat = np.array(mat)  # shape (samples, sequence_length)
    t=t[sequence_length:]
#    print("whole data shape  : ", whole_data.shape)

    X_an_find = mat[:, :-1]
    Y_an_find = mat[:, -1]
   
    X_an_find_2 = np.reshape(X_an_find, (X_an_find.shape[0], X_an_find.shape[1], 1))

    
    
    full_data_pred=model.predict(X_an_find_2)
#    print("full_data_pred data shape  : ", full_data_pred.shape)
#    print("Y_an_find data shape  : ", Y_an_find.shape)
    for i in range(0,len(full_data_pred)):
        if (full_data_pred[i]-Y_an_find[i])**2>(Y_an_find.max()/2)**2:
            anmly_pred.append(t[i-1])
    Accurate_predictions=np.intersect1d(anmly,anmly_pred)
    ####################################################
#    loss = history.history['loss']
#    val_loss = history.history['val_loss']
#    epochs2 = range(1, len(loss) + 1)
#    plt.plot(epochs2, loss, 'bo', label='Training loss')
#    plt.plot(epochs2, val_loss, 'b', label='Validation loss')
#    plt.title('Training and validation loss')
#    plt.xlabel('Epochs')
#    plt.ylabel('Loss')
#    plt.legend()
#    plt.show()
    ###################################################

    return model, t,Accurate_predictions,anmly_pred,anmly


model,t,Accurate_predictions,anmly_predicted,Actual_anmly=run_network()

