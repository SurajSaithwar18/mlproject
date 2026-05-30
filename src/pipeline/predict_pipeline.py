import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):

        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            print("Loading model...", flush=True)

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            print("Input Features:", flush=True)
            print(features, flush=True)

            transformed_data = preprocessor.transform(features)

            print("Transformation Successful", flush=True)

            preds = model.predict(transformed_data)

            print("Prediction Successful", flush=True)

            return preds

        except Exception as e:
            logging.error("Error occurred during prediction")
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender,
        race_ethnicity,
        parental_level_of_education,
        lunch,
        test_preparation_course,
        reading_score,
        writing_score
    ):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):

        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }

            df = pd.DataFrame(custom_data_input_dict)

            print("Created DataFrame:", flush=True)
            print(df, flush=True)

            return df

        except Exception as e:
            logging.error("Error occurred while creating DataFrame")
            raise CustomException(e, sys)