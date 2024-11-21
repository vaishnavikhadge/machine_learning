import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, numerical_columns, categorical_columns):
        """
        Creates preprocessing pipelines for numerical and categorical features.
        """
        try:
            # Numerical features pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical features pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))  # Scaling after OneHotEncoding
                ]
            )

            # Combine pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            logging.info("Preprocessing pipelines created.")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Transforms the training and testing datasets using the preprocessing pipeline.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("read th train and test data")

            
            train_df.columns = train_df.columns.str.strip().str.lower().str.replace(' ', '_')
            test_df.columns = test_df.columns.str.strip().str.lower().str.replace(' ', '_')

            train_df.drop(['rownumber', 'customerid', 'surname'], axis=1, inplace=True)
            test_df.drop(['rownumber', 'customerid', 'surname'], axis=1, inplace=True)
            logging.info("data cleaning the  done")

            
            target_column_name = "exited"
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            
            numerical_columns = [
                'creditscore', 'age', 'balance', 'numofproducts', 'hascrcard',
                'isactivemember', 'estimatedsalary', 'tenure', 'complain',
                'satisfaction_score', 'point_earned'
            ]
            categorical_columns = ['geography', 'gender', 'card_type']
            logging.info("preprocessing obj creating..")

            
            preprocessing_obj = self.get_data_transformer_object(numerical_columns, categorical_columns)

           
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info("combining features and target...")

            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("saving preprocessing object")

            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Data transformation completed.")
            return train_arr, test_arr, self.transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
