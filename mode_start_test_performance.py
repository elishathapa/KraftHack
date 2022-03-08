# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 06:57:14 2022

@author: Hai
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from matplotlib import pyplot
from sklearn.model_selection import train_test_split


url='C:/Users/Hai/Desktop/USN/Semester 4/ML competition/New folder/input_dataset-2.parquet'
df = pd.read_parquet(url)
start=df.loc[df['mode']=='start']
#Create base ML Model
data1=start[['Unit_4_Power','Unit_4_Reactive Power','Turbine_Pressure Drafttube','Turbine_Pressure Spiral Casing','Turbine_Rotational Speed' ]]
x1=data1.iloc[:,:].fillna(method='ffill')


    
data1=start[['Unit_4_Power','Unit_4_Reactive Power','Turbine_Pressure Drafttube','Turbine_Pressure Spiral Casing','Turbine_Rotational Speed' ]]
x1=data1.iloc[:,:].fillna(method='ffill')
y1=start['Bolt_4_Torsion'].fillna(method = 'ffill')
model1=LinearRegression()
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.33, random_state=1)
model1.fit(x1_train,y1_train)
y_pred1=model1.predict(x1_test)
a=mean_squared_error(y1_test,y_pred1)
print('the mean squared error for predicton model of bolt 4 torsion is',a)

y2=start['Bolt_6_Torsion'].fillna(method = 'ffill')
model2=LinearRegression()
x2_train, x2_test, y2_train, y2_test = train_test_split(x1, y2, test_size=0.33, random_state=1)
model2.fit(x2_train,y2_train)
y_pred2=model2.predict(x2_test)
a=mean_squared_error(y2_test,y_pred2)
print('the mean squared error for predicton model of bolt 6 torsion is',a)


data2=start[['Unit_4_Power','Unit_4_Reactive Power','Turbine_Rotational Speed' ]]
x3=data2.iloc[:,:].fillna(method='ffill')
y_pred1=model1.predict(x1)
y_pred2=model2.predict(x1)
x3['y1']=y_pred1
x3['y2']=y_pred2

y3=start['Bolt_1_Tensile'].fillna(method = 'ffill')
model3=LinearRegression()
x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y3, test_size=0.33, random_state=1)
model3.fit(x3_train,y3_train)
y_pred3=model3.predict(x3_test)
a=mean_squared_error(y3_test,y_pred3)
print('the mean squared error for predicton model of bolt 1 tensile is',a)

y4=start['Bolt_2_Tensile'].fillna(method = 'ffill')
model4=LinearRegression()
x4_train, x4_test, y4_train, y4_test = train_test_split(x3, y4, test_size=0.33, random_state=1)
model4.fit(x4_train,y4_train)
y_pred4=model4.predict(x4_test)
a=mean_squared_error(y4_test,y_pred4)
print('the mean squared error for predicton model of bolt 2 tensile is',a)

y5=start['Bolt_3_Tensile'].fillna(method = 'ffill')
model5=LinearRegression()
x5_train, x5_test, y5_train, y5_test = train_test_split(x3, y5, test_size=0.33, random_state=1)
model5.fit(x5_train,y5_train)
y_pred5=model5.predict(x5_test)
a=mean_squared_error(y5_test,y_pred5)
print('the mean squared error for predicton model of bolt 3 tensile is',a)

y6=start['Bolt_4_Tensile'].fillna(method = 'ffill')
model6=LinearRegression()
x6_train, x6_test, y6_train, y6_test = train_test_split(x3, y6, test_size=0.33, random_state=1)
model6.fit(x6_train,y6_train)
y_pred6=model6.predict(x6_test)
a=mean_squared_error(y6_test,y_pred6)
print('the mean squared error for predicton model of bolt 4 tensile is',a)

y7=start['Bolt_5_Tensile'].fillna(method = 'ffill')
model7=LinearRegression()
x7_train, x7_test, y7_train, y7_test = train_test_split(x3, y7, test_size=0.33, random_state=1)
model7.fit(x7_train,y7_train)
y_pred7=model7.predict(x7_test)
a=mean_squared_error(y7_test,y_pred7)
print('the mean squared error for predicton model of bolt 5 tensile is',a)

y8=start['Bolt_5_Tensile'].fillna(method = 'ffill')
model8=LinearRegression()
x8_train, x8_test, y8_train, y8_test = train_test_split(x3, y8, test_size=0.33, random_state=1)
model8.fit(x8_train,y8_train)
y_pred8=model8.predict(x8_test)
a=mean_squared_error(y8_test,y_pred8)
print('the mean squared error for predicton model of bolt 6 tensile is',a)