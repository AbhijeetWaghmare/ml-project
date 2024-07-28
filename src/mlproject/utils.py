import os
import sys

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging

import pandas as pd
import pymysql
from dotenv import load_dotenv

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
        logging.info(f"Connected to {mydb}")
        df = pd.read_sql_query("SELECT * from Students",mydb)
        print(df.head())
        return df
        
    except Exception as e:
        raise CustomException(e,sys)
