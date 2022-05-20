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
