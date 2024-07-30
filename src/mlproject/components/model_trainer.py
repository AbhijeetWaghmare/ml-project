from sklearn.linear_model import Lasso, LinearRegression
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from dataclasses import dataclass
import os
import sys
import numpy as np
from urllib.parse import urlparse


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

# Ridge, Lasso
# from sklearn.model_selection import RandomizedSearchCV

# from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.mlproject.utils import save_object, evaluate_models
import mlflow
from mlflow.models.signature import infer_signature
from dotenv import load_dotenv

import dagshub


load_dotenv()

mlflow_url = os.getenv("MLFLOW_URL")
repo_owner = os.getenv('REPO_OWNER')
repo_name = os.getenv('REPO_NAME')
dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig

    def evalution_model(self, true, predicted):
        mae = mean_absolute_error(true, predicted)
        mse = mean_squared_error(true, predicted)
        mse = np.sqrt(mse)
        r2_square = r2_score(true, predicted)
        return mse, mae, r2_square

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("split train test data")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )
            models = {
                "Linear Regression": LinearRegression(),
                # "Lasso": Lasso(),
                # "Ridge": Ridge(),
                # "KNeighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                # "CatBoostRegressor":CatBoostRegressor(verbose=True),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params = {
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                # "Gradient Boosting": {
                #     # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                #     "learning_rate": [0.1, 0.01, 0.05, 0.001],
                #     "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                #     # 'criterion':['squared_error', 'friedman_mse'],
                #     # 'max_features':['auto','sqrt','log2'],
                #     "n_estimators": [8, 16, 32, 64, 128, 256],
                # },
                "Linear Regression": {},
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                # "CatBoosting Regressor":{
                #     'depth': [6,8,10],
                #     'learning_rate': [0.01, 0.05, 0.1],
                #     'iterations': [30, 50, 100]
                # },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    # 'loss':['linear','square','exponential'],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            model_report: dict = evaluate_models(
                X_train, y_train, X_test, y_test, models, params
            )

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            models_names = list(params.keys())

            actual_model = ""

            for model in models_names:
                if best_model_name == model:
                    actual_model = actual_model + model
            print(actual_model)
            best_params = params[actual_model]

            # mlflow
            mlflow.set_registry_uri(mlflow_url)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                predicted_qualities = best_model.predict(X_test)

                rmse, mae, r2 = self.evalution_model(y_test, predicted_qualities)

                mlflow.log_params(best_params)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)

                # signature = infer_signature(X_train,predicted_qualities)
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(
                        best_model,
                        "model",
                        registered_model_name=actual_model,
                    )
                else:
                    mlflow.sklearn.log_model(best_model, "model")

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info("Best found model on training and test dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
