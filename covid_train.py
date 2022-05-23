# -*- coding: utf-8 -*-
"""
Created on Fri May 20 09:31:45 2022

This script is about the training of deep learning model using LSTM neural 
network to predict new cases in Malaysia using the past 30 days of number of 
cases.

cases_malaysia.csv = daily recorded covid-19 cases at country level

@author: User
"""

# packages
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
import datetime
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model
from sklearn.metrics import mean_absolute_error
from covid_modules import ModelCreation,DeploymentVisualization

#%% static code
DATASET_TRAIN_PATH = os.path.join(os.getcwd(),'data','cases_malaysia_train.csv')
DATASET_TEST_PATH = os.path.join(os.getcwd(),'data','cases_malaysia_test.csv')
SCALER_SAVE_PATH = os.path.join(os.getcwd(),'mms_scaler.pkl')# save scaler path
LOG_SAVE_PATH = os.path.join(os.getcwd(),'log_covid') 
log_covid = os.path.join(LOG_SAVE_PATH,
                                         datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model_covid.h5') #save model

#%% EDA
# 1) Load data
# 1) Data loading
X_train= pd.read_csv(DATASET_TRAIN_PATH)
X_test= pd.read_csv(DATASET_TEST_PATH)

# 2) Data inspection/visualization
X_train.info()
# date and cases new are object
# convert new cases column to numeric data
X_train['cases_new']= pd.to_numeric(X_train['cases_new'],errors='coerce')
X_train.describe().T
X_train.isnull().sum() 
# 12 mising data for cases_new column
# 342 missing information for all the cluster columns

X_test.info()
# no need to change data type for cases_new column as previously
X_test.describe().T
# missing one data for cases_new column
X_test.isnull().sum() 


# 3) Data cleaning
# deal with missing data
# replace missing data using interpolation
# since only want to use cases_new column, fill NaN for that column only
plt.plot(X_train['cases_new'],color='blue') # before interpolation
plt.show()
X_train['cases_new']=X_train['cases_new'].interpolate(method='polynomial',order=3)
plt.plot(X_train['cases_new'],color='red') # after interpolation
plt.show()
# pattern of graph still the same and did not looks weird with some points shooting up
# recheck null value
X_train['cases_new'].isnull().sum()
# no more missing data for cases new column

# replace missing data using interpolation for testing dataset as well
plt.plot(X_test['cases_new'],color='blue') # before interpolation
plt.show()
X_test['cases_new']=X_test['cases_new'].interpolate(method='polynomial',order=3)
plt.plot(X_test['cases_new'],color='red') # after interpolation
plt.show()
# pattern of graph still the same and did not looks weird with some points shooting up
# recheck data
X_test['cases_new'].isnull().sum()
# no more missing data for cases new column


# 4) Data selected
# extract data from cases_new column for prediction
x_train_input = X_train['cases_new'].values
x_test_input = X_test['cases_new'].values
# plot line graph
plt.figure()
plt.plot(x_train_input)
plt.show()
# there is some trend here
# 3 peaks observed
# some outliers is due to the missing data replaced


# 5) Data preprocessing
# usually use MinMaxScalar because time series data normally don't have -ve
mms = MinMaxScaler() 
x_train_scaled = mms.fit_transform(np.expand_dims(x_train_input,-1))
# only use fit for training dataset
x_test_scaled = mms.transform(np.expand_dims(x_test_input,-1))
# save scaler to pickle file
mms_scaler='mms.pkl'
pickle.dump(mms,open(mms_scaler,'wb'))


# prepare training data to load into model
x_train = [] #creat empty list so we can append smthg to it
y_train = []
# prepare training dataset with window size 30 days
window_size = 30

for i in range(window_size,len(x_train_scaled)): #(30,680)
    x_train.append(x_train_scaled[i-window_size:i,0]) #size=650
    y_train.append(x_train_scaled[i,0])#value predicted after go through 30 datapoints

x_train = np.array(x_train)
x_train = np.expand_dims(x_train,axis=-1) #expand dimension
y_train = np.array(y_train)


dataset_total= np.concatenate((x_train_scaled,x_test_scaled),axis=0) #(780,1)
# we have to concatenate first because we do the training with 30 datapoints
# training dataset consist of 100 data points

# prepare testing data to load into model
x_test= []
y_test= []

#USE data or temp
# get the last 680 or get the last 130??
window_size = 30
length_window = window_size+len(x_test_scaled) #30+100=130
temp = dataset_total[-length_window:] # get the last 130 numbers

for i in range(window_size,len(temp)):#(30,130)
    x_test.append(temp[i-window_size:i,0])
    y_test.append(temp[i,0])

x_test = np.array(x_test)
x_test = np.expand_dims(x_test,axis=-1) 
y_test = np.array(y_test)

#%% Model creation
mc = ModelCreation()
model = mc.model_layers(nodes=64,datashape=x_train.shape,n=1)
model.summary()
plot_model(model)
model.compile(optimizer='adam',loss='mse',metrics='mse')

#%% View training loss
# callbacks (tensorboard) for visualization
tensorboard_callback = TensorBoard(log_dir= log_covid, histogram_freq=1)

hist = model.fit(x_train,y_train,epochs=100,
                  callbacks=[tensorboard_callback])

#%% Save model
model.save(MODEL_SAVE_PATH)

#%% Deployment for prediction
predicted=[]
for i in x_test:
    predicted.append(model.predict(np.expand_dims(i,axis=0)))
predicted = np.array(predicted)

# use inverse transform to get back the ori value
# because previously we applied min max scaler
inversed_y_true = mms.inverse_transform(np.expand_dims(y_test,axis=-1))
inversed_y_predict = mms.inverse_transform(np.array(predicted).reshape(len(predicted),1))

# graph
v= DeploymentVisualization()
v.predict_actual_graph(predict=inversed_y_predict, true=inversed_y_true)

#%% Performance evaluation
y_true = inversed_y_true
y_pred = inversed_y_predict
# y true is our y test
print((mean_absolute_error(y_true,y_pred)/sum(abs(y_true)))*100)
# reult shown mape obtained is 0.142
# Low mean absolute percentage error indicate the forecasting system
# can predict accurately

