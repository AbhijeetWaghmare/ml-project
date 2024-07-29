import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import os
from src.mlproject.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """This function is for data transformation"""
        try:
            numerical_columns = ["reading score", "writing score"]

            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]

            num_pipline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info(f"Categorial columns{categorical_columns}")
            logging.info(f"Numerical columns{numerical_columns}")

            # numeric_transformer = StandardScaler()
            # oh_transformer = OneHotEncoder()

            preprocessor = ColumnTransformer(
                [
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                    ("num_pipeline", num_pipline, numerical_columns),
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading the train and test file")

            preprocessor_object = self.get_data_transformer_object()
            
            
            target_column_name = "math score"
            numerical_columns = ["reading score", "writing score"]

            #divide train dataset
            input_features_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            #divide test dataset
            input_features_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            # appling preprocessing on train and test dataset
            logging.info("Applying preprocessing on train and test dataset")
            input_features_train_arr = preprocessor_object.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessor_object.transform(input_features_test_df)
            
            train_arr = np.c_[input_features_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_features_test_arr,np.array(target_feature_test_df)]
            
            logging.info(f"Save preprocessing object")
            
            save_object(self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessor_object)
            
            return (
                train_arr,test_arr, self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
