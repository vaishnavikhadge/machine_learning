import sys
from src.logger import logging

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        # Correct use of super() to call parent class constructor
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_detail)

    def get_detailed_error_message(self, error_message, error_detail: sys):
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return f"Error occurred in script: {file_name}, line: {line_number}, message: {error_message}"

    def __str__(self):
        return self.error_message



