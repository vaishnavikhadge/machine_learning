import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
import pandas as pd  
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from lightgbm import LGBMRegressor  # Import LightGBM Regressor

@dataclass
class ModelTrainConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrain:
    def __init__(self):
        self.model_train_config = ModelTrainConfig()

    def initiate_model_train(self, train_array, test_array):
        try:
            logging.info("Split training and test input data.")
            X_train, Y_train = train_array[:, :-1], train_array[:, -1]
            X_test, Y_test = test_array[:, :-1], test_array[:, -1]

            # Define the models to evaluate, including LightGBM
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor(),
                "LightGBM": LGBMRegressor()  # Add LightGBM model here
            }

            # Evaluate the models
            model_report = evaluate_models(X_train=X_train, X_test=X_test, y_train=Y_train, y_test=Y_test, models=models)
            
            # Get the best model
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.info("No best model found with acceptable performance.")
                return None  # If no model performs well

            logging.info(f"Best model found: {best_model_name} with a score of {best_model_score}")

            # Save the best model
            save_object(
                file_path=self.model_train_config.trained_model_file_path,
                obj=best_model
            )

            # Make predictions using the best model
            predicted = best_model.predict(X_test)

            # Calculate R^2 score
            r2_square = r2_score(Y_test, predicted)
            logging.info(f"R^2 Score on test data: {r2_square}")
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
