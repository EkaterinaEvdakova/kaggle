import streamlit as st
#import os 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
#import joblib 
from joblib import  load
from datetime import date
#from sklearn.preprocessing import StandardScaler
import lightgbm
import re
#from lightgbm import LGBMRegressor


#### Функции
#Функция для проверки пустых срок
def checkNaN(str):
    return str != str

def to_lower(data):    
    data['body'] = data['body'].apply(clear_text)
    data['make'] = data['make'].apply(clear_text)
    data['model'] = data['model'].apply(clear_text)
    data['seller'] = data['seller'].apply(clear_text)
    data['trim'] = data['trim'].apply(clear_text)
    return data

def data_no_color(data):
    data2 = data.copy()
    data2['color'] = data2['color'].astype(str)
    data2['color'] = data2['color'].where(
    data2['color']=='white',1)
    data2['color'] = data2['color'].where(
    data2['color']==1,0)
    data2['color'] = data2['color'].astype(int)
    data2 = data2.drop(['year_sale'],axis = 1)   
    return data2

def year_month_age(data):
  data['year_sale'] = pd.to_datetime(data['saledate'], utc = True).dt.year
  data['month_sale'] = pd.to_datetime(data['saledate'], utc = True).dt.month
  data["age_in_days"] = (data['year_sale'] - data['year']) * 365 + pd.to_datetime(
      data['saledate'], utc = True).dt.day
  return data

# функция для отчистки текста
def clear_text(text):
    text_clear = re.sub(r"[^a-zA-Z]", ' ' ,  text)
    text_clear = re.sub(r'[^\w\s]', ' ', text).strip()
    text_clear = " ".join(text_clear.split())
    text_clear = text_clear.lower()
    return text_clear

def prepare_data(data):
    data = to_lower(data)
    data['color'] = data['color'].fillna('-')
    data['condition'] =data['condition'].astype('float')
    data['odometer'] =data['odometer'].astype('int')
    data['year_sale'] = date.today().year
    data['month_sale'] = date.today().month
    data["age_in_days"] = (date.today().year - data['year']) * 365 + date.today().day
    data['mileage_per_day'] = data['odometer']/data['age_in_days']
    #Масштабирование численных признаков
    num_features = data.select_dtypes(include=np.number).columns
    scaler =load('std_scaler.save')
    data[num_features] = scaler.transform(data[num_features])
    features_ohe = list(set(data.columns) - set(num_features))
    for c in features_ohe:
        data[c] = data[c].astype('category')
    
    data = data_no_color(data)
    data = data.drop(columns = [ 'age_in_days'], axis = 1)   
    return data 

## Загрузка данных
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train =train.dropna()
train_after = to_lower(train)

years = np.unique(train_after.year)
brand = np.unique(train_after.make.astype('str'))
model = np.unique(train_after.model.astype('str'))
odometer = list(range(1000, 400000, 5000))
condition = np.unique(train_after.condition.astype('str'))
color = np.unique(train_after.color.astype('str'))
seller = np.unique(train_after.seller.astype('str'))
body =np.unique(train_after.body.astype('str'))
transmission = np.unique(train_after.transmission.astype('str'))
trim = np.unique(train_after.trim.astype('str'))
country = ['United States', 'Canada', 'South Korea', 'Sweden', 'Germany',
       'Japan', 'Mexico', 'United Kingdom', 'Turkey', 'Hungary', 'Italy',
       'Australia', 'Thailand', 'Brazil', 'France', 'Finland']
state = np.unique(train_after.state.astype('str'))


st.title('Оценка стоимости автомобиля')
st.header('Онлайн-калькулятор стоимости автомобиля')
st.sidebar.markdown("## Параметры автомобиля")
select_event_2 = st.sidebar.selectbox(
    'Марка', brand.tolist())
select_event_9 = st.sidebar.selectbox(
    'Модель', model.tolist())
select_event = st.sidebar.selectbox(
    'Год выпуска', years.tolist())
select_event_3 = st.sidebar.selectbox(
    'Пробег', odometer)
select_event_4 = st.sidebar.selectbox(
    'Состояние машины', condition)
select_event_5 = st.sidebar.selectbox(
    'Цвет машины', color)
select_event_6 = st.sidebar.selectbox(
    'Продавец', seller)
select_event_7 = st.sidebar.selectbox(
    'transmission', transmission)
select_event_8 = st.sidebar.selectbox(
    'trim', trim)
select_event_10 = st.sidebar.selectbox(
    'body', body)
select_event_13 = st.sidebar.selectbox(
    'state', state)
select_event_11 = st.sidebar.selectbox(
    'seller', seller)
select_event_12 = st.sidebar.selectbox(
    'country', country)


test = [select_event, select_event_2, select_event_9, select_event_8, 
        select_event_10, select_event_7, select_event_13,
        select_event_4, select_event_3, select_event_5,  select_event_11,
        select_event_12]
df_test = pd.DataFrame([test], columns = ['year', 'make', 'model', 
                                        'trim', 'body', 'transmission', 
                                        'state','condition', 
                                        'odometer', 'color',
                                        'seller', 'country'])

st.header('Статистика: ' + str(select_event_2))
data = train_after[train_after.make == select_event_2]
stat_make = round(data.describe(), 3)
st.write(stat_make)

fig1, ax =plt.subplots()
ax = sns.scatterplot(data =data, x = 'year', y = 'sellingprice')

fig2, ax =plt.subplots()
ax = sns.scatterplot(data =data, x = 'odometer', y = 'sellingprice')

fig3, ax =plt.subplots()
ax = sns.scatterplot(data  =data, x = 'condition', y = 'sellingprice')

fig = px.scatter(
    data ,
    x="year",
    y="sellingprice",
    size="condition",
    color="model",
    hover_name="seller",
    log_x=True,
    size_max=60,
)


tab1, tab2, tab3, tab4 = st.tabs(["Цена/модель/год", "Год / cтоимость",
                      "Пробег/ год", 
                      'Состояние/ год'])

with tab1:
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
with tab2:
    st.pyplot(fig1)
with tab3:
    st.pyplot(fig2)
with tab4:
    st.pyplot(fig3)

with st.sidebar:
    if st.button('Дать прогноз'):
        df_test = prepare_data(df_test)
        model = lightgbm.Booster(model_file='model_lgbm_best3.txt')
        price = abs(model.predict(df_test))
        st.write(round(price[0]), '$')
    else:
        st.write('Нажмите для прогноза')