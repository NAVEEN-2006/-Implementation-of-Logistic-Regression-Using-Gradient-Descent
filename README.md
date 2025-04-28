## Implementation-of-Logistic-Regression-Using-Gradient-Descent


## AIM:


To write a program to implement the the Logistic Regression Using Gradient Descent.


## Equipments Required:


1.Hardware – PCs


2.Anaconda – Python 3.7 Installation / Jupyter notebook


## Algorithm


1.Import the required libraries.


2.Load the dataset.


3.Define X and Y array.


4.Define a function for costFunction,cost and gradient.


5.Define a function to plot the decision boundary. 6.Define a function to predict the Regression value.


## Program:

Program to implement the the Logistic Regression Using Gradient Descent.


Developed by: NAVEEN KUMAR S


RegisterNumber:  212223040129

```

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
data=pd.read_csv("placement.csv")  
data=data.drop('sl_no',axis=1) 
data=data.drop('salary',axis=1) 
data
data["gender"]=data["gender"].astype('category') 
data["ssc_b"]=data["ssc_b"].astype('category') 
data["hsc_b"]=data["hsc_b"].astype('category') 
data["degree_t"]=data["degree_t"].astype('category') 
data["workex"]=data["workex"].astype('category') 
data["specialisation"]=data["specialisation"].astype('category') 
data["status"]=data["status"].astype('category') 
data["hsc_s"]=data["hsc_s"].astype('category') 
data.dtypes
data["gender"]=data["gender"].cat.codes 
data["ssc_b"]=data["ssc_b"].cat.codes 
data["hsc_b"]=data["hsc_b"].cat. codes
data["degree_t"]=data["degree_t"].cat.codes 
data["workex"]=data["workex"].cat.codes 
data["specialisation"]=data["specialisation"].cat.codes 
data["status"]=data["status"].cat.codes 
data["hsc_s"]=data["hsc_s"].cat.codes 
data 
x=data.iloc[:,:-1].values 
y=data.iloc[:,-1].values
y 
theta = np.random.randn(x.shape[1]) 
Y=y 
def sigmoid(z): 
   return 1/(1+np.exp(-z))
def loss(theta,X,y): 
   h=sigmoid(X.dot(theta))
   return -np.sum(y*np.log(h)+(1-y)*np.log(1-h)) 
def gradient_descent(theta,X,y,alpha,num_iterations): 
    m=len(y)
    for i in range(num_iterations): 
        h=sigmoid(X.dot(theta)) 
        gradient = X.T.dot(h-y)/m 
        theta-=alpha * gradient 
        return theta
gradient_descent(theta,x,y,alpha=0.01,num_iterations=1000) 
def predict(theta,X): 
    h=sigmoid(X.dot(theta)) 
    y_pred=np.where(h>=0.5,1,0) 
    return y_pred 
y_pred = predict(theta,x) 
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy: ",accuracy) 
print(y_pred)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]]) 
y_prednew=predict(theta,xnew) 
print(y_prednew) 
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]]) 
y_prednew=predict(theta,xnew) 
print(y_prednew)

```

## Output:


## Dataset :


![image](https://github.com/user-attachments/assets/a02a0d73-5069-4ed7-ac77-7486d42604f4)



## Information :


![image](https://github.com/user-attachments/assets/65db5dec-0adf-4af7-9da7-c075b4e2ce24)


## Encoding:


![image](https://github.com/user-attachments/assets/6ba4c064-306b-4227-84d1-b1c5e9d26502)



## X and Y value:


![image](https://github.com/user-attachments/assets/d92ddf36-b60f-48a7-b988-56df0e612a13)



![image](https://github.com/user-attachments/assets/f4bd7cb1-aced-4340-9610-2b2c52d28734)


## Gradient Descent:


![image](https://github.com/user-attachments/assets/d588f746-5455-4bcb-b98f-e5ebe136597b)



## Accuracy:


![image](https://github.com/user-attachments/assets/8728e3fc-03af-4636-b6ca-ab31b5c7026a)



## Prediction:

![image](https://github.com/user-attachments/assets/3a03492b-4e8c-4d2d-bfca-aa7201088d6a)


![image](https://github.com/user-attachments/assets/44c34ce9-36d9-4732-8354-dafa38fadf60)


## Result:


Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
