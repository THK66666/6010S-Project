import os
os.chdir(r'C:\Users\haokaitang\Desktop\data')
from sklearn import metrics, ensemble
from sklearn.model_selection import train_test_split
import xgboost as xgb
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import os
warnings.filterwarnings('ignore')
plt.style.use('ggplot')

df = pd.read_csv('train.csv')  
df = df.sample(frac=0.1)  
songs = pd.read_csv('songs.csv')
df = pd.merge(df, songs, on='song_id', how='left')
members = pd.read_csv('members.csv')
df = pd.merge(df, members, on='msno', how='left')  

##According to the data type, select the corresponding column in df
for i in df.select_dtypes(include=['object']).columns: 
    df[i][df[i].isnull()] = 'unknown'  
##Fill the NAN data with 0
df = df.fillna(value=0)  


##The data of "registration_init_time" and "expiration_date_year" in the "members" csv file is 20140720 20171026. 
##Convert them into year, month and day format, and get the corresponding year, month and day respectively.
df.registration_init_time = pd.to_datetime(df.registration_init_time, format='%Y%m%d', errors='ignore')  
df['registration_init_time_year'] = df['registration_init_time'].dt.year 
df['registration_init_time_month'] = df['registration_init_time'].dt.month 
df['registration_init_time_day'] = df['registration_init_time'].dt.day  

df.expiration_date = pd.to_datetime(df.expiration_date,  format='%Y%m%d', errors='ignore')
df['expiration_date_year'] = df['expiration_date'].dt.year
df['expiration_date_month'] = df['expiration_date'].dt.month
df['expiration_date_day'] = df['expiration_date'].dt.day

##Select the columns in df, use columns_name to select the columns you need
df = df[['msno', 'song_id', 'source_screen_name', 'source_type', 'target',
       'song_length', 'artist_name', 'composer', 'bd',
       'registration_init_time', 'registration_init_time_month',
       'registration_init_time_day', 'expiration_date_day']]

##Set the data type of a column in df, because in the subsequent text digitization process, only ‘category’ type data can be digitized by .cat.codes function
df['registration_init_time'] = df['registration_init_time'].astype('category')
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category')
for col in df.select_dtypes(include=['category']).columns:
    df[col] = df[col].cat.codes
##Extract the label, delete the "target" column in df, and assign the pop-up "target" to the target as the label of the sample
target = df.pop('target')  
train_data, test_data, train_labels, test_labels = train_test_split(df, target, test_size = 0.3)



##The function of cat.codes function is to digitally convert non-numeric objects, for example: male and female conversion results are: 0, 1; high, medium and low conversion results are: 0, 1, 2 (text Digitizing)
model = CatBoostClassifier(learning_rate=0.1, depth=10, iterations=300,)
model.fit(train_data, train_labels)
##Predict
predict_labels = model.predict(test_data)
##Enter evaluation indicators, precision/ recall /f1-score support
print('CatBoost Result：')
print(metrics.classification_report(test_labels, predict_labels))  


##Random Forest
model1 = ensemble.RandomForestClassifier(n_estimators=350, max_depth=40)
model1.fit(train_data, train_labels)
predict_labels1 = model1.predict(test_data)
print('RandoForest Result：')
print(metrics.classification_report(test_labels, predict_labels1))


##XGBoost
model2 = xgb.XGBClassifier(learning_rate=0.1, max_depth=10, min_child_weight=10, n_estimators=250)
model2.fit(train_data, train_labels)
predict_labels2 = model2.predict(test_data)
print('Xgboost Result：')
print(metrics.classification_report(test_labels, predict_labels2))