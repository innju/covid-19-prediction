# -*- coding: utf-8 -*-
"""
Created on Fri May 20 09:32:05 2022

This script consist of the classes and functions to be used in covid_train.py

@author: User
"""

# packages
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

#%% Classes and functions

class ModelCreation():
    def __init__(self):
        pass
    
    def model_layers(self,nodes,datashape,n):
        '''
        Thsi function creates deep learning model layers.

        Parameters
        ----------
        nodes : number
            number of nodes for LSTM layer of the model.
        datashape : size
            the shape of the training data.
        n : number
            select input shape by refer to the position in datashape returned.

        Returns
        -------
        model : model
            deep learning model created.

        '''
        model = Sequential()
        model.add(LSTM(nodes,activation='tanh',
                        return_sequences= True,
                        input_shape=(datashape[n],1))) # input shape (30,1)
        model.add(Dropout(0.2)) # dropout layer
        model.add(LSTM(nodes)) # hidden layer
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='relu')) #because we have only one column
        
        return model
    
class DeploymentVisualization():
    def __init__(self):
        pass
    
    def predict_actual_graph(self,predict,true):
        plt.figure()
        plt.plot(predict, color='r')
        plt.plot(true, color='b')
        plt.legend(['predicted','actual'])
        plt.show()
        
        
