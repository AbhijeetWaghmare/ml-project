import os
import sys

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging

import pandas as pd
import pymysql
from dotenv import load_dotenv
import pickle
import numpy as np
from sklearn.metrics import r2_score 
from sklearn.model_selection import GridSearchCV


load_dotenv()

host = os.getenv("DATABASE_HOST")
db = os.getenv("DATABASE_NAME")
user = os.getenv("DATABASE_USER")
password = os.getenv("DATABASE_PASSWORD")


def read_sql_data():
    # os.environ.get('')
    logging.info("Reading sql databse started ")
    try:
        mydb = pymysql.connect(host=host,user=user,password=password,db=db)
        logging.info(f"Connected to db")
        df = pd.read_sql_query("SELECT * from Students",mydb)
        # print(df.head())
        return df
        
    except Exception as e:
        raise CustomException(e,sys)
    

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
            raise CustomException(e,sys)


def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)