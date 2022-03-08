
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from matplotlib import pyplot



url='C:/Users/Hai/Desktop/USN/Semester 4/ML competition/New folder/input_dataset-2.parquet'
df = pd.read_parquet(url)
start=df.loc[df['mode']=='start']
#Create base ML Model
data1=start[['Unit_4_Power','Unit_4_Reactive Power','Turbine_Pressure Drafttube','Turbine_Pressure Spiral Casing','Turbine_Rotational Speed' ]]
x1=data1.iloc[:,:].fillna(method='ffill')

y1=start['Bolt_4_Torsion'].fillna(method = 'ffill')

model1=LinearRegression()
model1.fit(x1,y1)
y_pred1=model1.predict(x1)

y2=start['Bolt_6_Torsion'].fillna(method = 'ffill')

model2=LinearRegression()
model2.fit(x1,y2)
y_pred2=model2.predict(x1)

data2=start[['Unit_4_Power','Unit_4_Reactive Power','Turbine_Rotational Speed' ]]
x3=data2.iloc[:,:].fillna(method='ffill')

x3['y1']=y_pred1
x3['y2']=y_pred2

y3=start['Bolt_1_Tensile'].fillna(method = 'ffill')
model3=LinearRegression()
model3.fit(x3,y3)

y4=start['Bolt_2_Tensile'].fillna(method = 'ffill')
model4=LinearRegression()
model4.fit(x3,y4)

y5=start['Bolt_3_Tensile'].fillna(method = 'ffill')
model5=LinearRegression()
model5.fit(x3,y5)

y6=start['Bolt_4_Tensile'].fillna(method = 'ffill')
model6=LinearRegression()
model6.fit(x3,y6)

y7=start['Bolt_5_Tensile'].fillna(method = 'ffill')
model7=LinearRegression()
model7.fit(x3,y7)

y8=start['Bolt_6_Tensile'].fillna(method = 'ffill')
model8=LinearRegression()
model8.fit(x3,y8)

#calculate results
url1='C:/Users/Hai/Desktop/USN/Semester 4/ML competition/New folder/prediction_input.parquet'
df = pd.read_parquet(url1)
start=df.loc[df['mode']=='start']
#Create base ML Model
data1=start[['Unit_4_Power','Unit_4_Reactive Power','Turbine_Pressure Drafttube','Turbine_Pressure Spiral Casing','Turbine_Rotational Speed' ]]
x1=data1.iloc[:,:].fillna(method='ffill')

y_pred1=model1.predict(x1)
y_pred2=model2.predict(x1)

data2=start[['Unit_4_Power','Unit_4_Reactive Power','Turbine_Rotational Speed' ]]
x3=data2.iloc[:,:].fillna(method='ffill')

x3['y1']=y_pred1
x3['y2']=y_pred2

#result
y_result1=model3.predict(x3)
y_result2=model4.predict(x3)
y_result3=model5.predict(x3)
y_result4=model6.predict(x3)
y_result5=model7.predict(x3)
y_result6=model8.predict(x3)



start['Bolt_1_Tensile']=y_result1
start['Bolt_2_Tensile']=y_result2
start['Bolt_3_Tensile']=y_result3
start['Bolt_4_Tensile']=y_result4
start['Bolt_5_Tensile']=y_result5
start['Bolt_6_Tensile']=y_result6
df=pd.DataFrame(start)
df.to_csv("Mode_start_result.csv") 