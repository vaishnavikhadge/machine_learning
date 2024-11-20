import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd  # type: ignore

from sklearn.model_selection import train_test_split  
from dataclasses import dataclass

# DataIngestionConfig to define file paths for train, test, and raw data
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()  # Use 'self' here, not 'Self'

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method")

        try:
            # Ensure the file path is correct
            df = pd.read_csv('notebook/data/archive (1).zip')
            logging.info("Read the dataset as a dataframe")

            # Create directories for artifacts
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved")

            # Train-test split
            logging.info("Train-test split started")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=5)

            # Save train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

# Ensure the __main__ block is outside the class
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
