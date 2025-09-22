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

<img width="401" height="283" alt="image" src="https://github.com/user-attachments/assets/ebdd5488-de0c-4d75-81ca-21c046d727d4" />


<img width="470" height="257" alt="image" src="https://github.com/user-attachments/assets/393bcab6-aee4-4bf9-a2a5-6b8c313da0b4" />


<img width="200" height="128" alt="image" src="https://github.com/user-attachments/assets/fe58481b-1880-4393-aac1-8825cfc5d099" />


<img width="350" height="281" alt="image" src="https://github.com/user-attachments/assets/71002a65-ad64-4945-85b4-c5d1431b50b3" />


<img width="431" height="113" alt="image" src="https://github.com/user-attachments/assets/8d2bad16-0205-4839-b604-0ebb6bf8240f" />


<img width="617" height="172" alt="image" src="https://github.com/user-attachments/assets/86d84f73-80fd-4a11-9a54-a4b80f33682e" />


<img width="1656" height="136" alt="image" src="https://github.com/user-attachments/assets/c6113362-5eb6-42c0-ac83-bdbdd6a954b2" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
