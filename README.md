![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)

# covid-19-prediction
Covid-19 prediction using deep learning model. The purpose is to predict the number of new cases for the next day by tellong the model to predict based on previous 30 days data.

The python scripts uploaded had been tested and run using Spyder(Python 3.8).
<br>The source of the data used for this analysis is:
<br>[Data source](https://github.com/MoH-Malaysia/covid19-public)
<br>It is official data on the COVID-19 epidemic in Malaysia. Powered by CPRC, CPRC Hospital System, MKAK, and MySejahtera.

### FOLDERS AND FILES
**data folder**: training and testing dataset
<br>**figures folder**: the graph of predicted and actual value, tensorboard interface for performance evaluation, model architecture
<br>**log_covid folder**: history of training for tensorboard visualization.
<br>**__pycache__**: Auto generated file to connect the modules with the training file.
<br>**init.py**: initial file to connect classes and functions with training file.
<br>**mms.pkl**: pickle file stored mean max scaler
<br>**model_covid.h5**: saved trained model
<br>**covid_modules.py**: all the classes and functions created to ease the training process
<br>**covid_train.py**: python script for model training


### MODEL
Figure below show the architecture of the model.

![Image](https://github.com/innju/covid-19-prediction/blob/main/figures/model.png)

<br>It consists of two LSTM layer and two dropout layer.Final dense layer consists of one node because the prediction is based on one single column only, which is cases_new. 


<br>The graph of predicted and actual value is shown as below.

![Image](https://github.com/innju/covid-19-prediction/blob/main/figures/predicted_vs_actual.png)

<br>The value predicted for the next day is 26344.6 which is equivalent to 26345 number of cases.Based on the graph displayed, the line for predicted data and the line for actual data have the similar pattern. This means the prediction model tend to  have good performance in making the prediction. This is further supported by the mean absolute percentage error(MAPE) calculated, which is only 0.137%. MAPE is a common metric used to measure the performance of the forecasting model.MAPE in this analysis indicate that the average difference between the forecasted value and the actual value is 0.137%. Therefore, you can see the predicted value is not exactly the same as the actual value.

<br>The figrue below show the performance for epoch loss and epoch mse. 

![Image](https://github.com/innju/covid-19-prediction/blob/main/figures/tensorboard.png)

<br>Both of the graphs showing decrease in loss and mse measured. It is good because low loss and low mse indicates the model can performed well. Since the loss and metric chosen to measure the performance are mse, they have the same pattern. Batch normalization is not introduced because it will caused bias, causing the model to perform better with earlier batch samples compared to the later batch samples. Batch normalization is not a ideal way to deal with when the dataset is in time series form.





Thanks for reading.
