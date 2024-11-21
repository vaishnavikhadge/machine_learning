import sys

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        self.error_message = self.get_detailed_error_message(error_message, error_detail)

    def get_detailed_error_message(self, error_message, error_detail: sys):
        exc_type, exc_value, exc_tb = error_detail.exc_info()  # Get error details

        if exc_tb is None:
            # If there is no traceback (e.g., no active exception), return a simple message
            return f"Error occurred: {error_message}"

        # Extract file name and line number from traceback if exc_tb is not None
        filename = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        return f"Error occurred in script: {filename}, line: {line_number}, message: {error_message}"
