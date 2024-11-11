# 1 : import libraries 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score


# 2 : read data 
data = load_breast_cancer()
x = data.data 
y = data.target 

# define variables 
random_state=12
test_size=0.2


# split 
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=test_size,random_state=random_state)

# define models dict 
models = {
    "RandomForest" : RandomForestClassifier(n_estimators=100,random_state=random_state) , 
    "LogisticRegression" : LogisticRegression(random_state=random_state),
    "KNN" : KNeighborsClassifier() , 
    "DecisionTree": DecisionTreeClassifier(random_state=random_state)
}


# function to train each model --> save accuracy
def train_and_evaluate_model(model_name , model , x_train , y_train , x_test , y_test):
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    
    # save model , accuracy in file 
    with open("metrices.txt",'a') as file:  # a=append 
        file.write(f"{model_name}\n")
        file.write(f"Accuracy : {accuracy}\n")
        file.write("-"*30+"\n")
    
    



for model_name , model in models.items():
    train_and_evaluate_model(model_name,model,x_train,y_train,x_test,y_test)