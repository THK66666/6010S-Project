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

for i in df.select_dtypes(include=['object']).columns: 
    df[i][df[i].isnull()] = 'unknown'  # 
df = df.fillna(value=0)  

df.registration_init_time = pd.to_datetime(df.registration_init_time, format='%Y%m%d', errors='ignore')  
df['registration_init_time_year'] = df['registration_init_time'].dt.year 
df['registration_init_time_month'] = df['registration_init_time'].dt.month 
df['registration_init_time_day'] = df['registration_init_time'].dt.day  

df.expiration_date = pd.to_datetime(df.expiration_date,  format='%Y%m%d', errors='ignore')
df['expiration_date_year'] = df['expiration_date'].dt.year
df['expiration_date_month'] = df['expiration_date'].dt.month
df['expiration_date_day'] = df['expiration_date'].dt.day

df = df[['msno', 'song_id', 'source_screen_name', 'source_type', 'target',
       'song_length', 'artist_name', 'composer', 'bd',
       'registration_init_time', 'registration_init_time_month',
       'registration_init_time_day', 'expiration_date_day']]

df['registration_init_time'] = df['registration_init_time'].astype('category')
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category')
for col in df.select_dtypes(include=['category']).columns:
    df[col] = df[col].cat.codes
target = df.pop('target')  
train_data, test_data, train_labels, test_labels = train_test_split(df, target, test_size = 0.3)




model = CatBoostClassifier(learning_rate=0.1, depth=10, iterations=300,)
model.fit(train_data, train_labels)
predict_labels = model.predict(test_data)
print('CatBoost Result：')
print(metrics.classification_report(test_labels, predict_labels))  


model1 = ensemble.RandomForestClassifier(n_estimators=350, max_depth=40)
model1.fit(train_data, train_labels)
predict_labels1 = model1.predict(test_data)
print('RandoForest Result：')
print(metrics.classification_report(test_labels, predict_labels1))

model2 = xgb.XGBClassifier(learning_rate=0.1, max_depth=10, min_child_weight=10, n_estimators=250)
model2.fit(train_data, train_labels)
predict_labels2 = model2.predict(test_data)
print('Xgboost Result：')
print(metrics.classification_report(test_labels, predict_labels2))