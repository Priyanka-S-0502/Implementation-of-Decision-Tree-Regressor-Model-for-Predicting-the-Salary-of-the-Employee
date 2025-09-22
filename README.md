# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
 
2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.Calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: PRIYANKA S
RegisterNumber: 212224040255 
*/

import pandas as pd
import numpy as np
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
lr=LabelEncoder()
data["Position"]=lr.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
rmse = np.sqrt(mse)
print("REG-NO:212224040255")
print("NAME:PRIYANKA S")
print("Mean Squared Error:",mse)
print("Mean absolute Error:",mae)
print("R2 score:",r2)
print("Root Mean Square Error:",rmse)
dt.predict([[5,6]])
```

## Output:

<img width="1629" height="280" alt="image" src="https://github.com/user-attachments/assets/4f1c914d-79c4-42d9-bf8f-94c3d53b8010" />

<img width="1608" height="262" alt="image" src="https://github.com/user-attachments/assets/b0d92761-cca5-4fdd-adca-34a21200de79" />

<img width="1636" height="114" alt="image" src="https://github.com/user-attachments/assets/0ed0d73d-a6dd-4a85-80c6-5876d487977f" />

<img width="1636" height="287" alt="image" src="https://github.com/user-attachments/assets/4143a234-3625-43fb-abec-3747e3384832" />

<img width="1635" height="129" alt="image" src="https://github.com/user-attachments/assets/e3f06b18-5cc2-4acf-a8e9-06d9c46aeecd" />

<img width="1643" height="169" alt="image" src="https://github.com/user-attachments/assets/eead10ed-b025-44d4-ba5b-43ef5ab122da" />

<img width="1664" height="125" alt="image" src="https://github.com/user-attachments/assets/3b8a4b39-0702-42bf-9db2-3f50b6199e43" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
