import json
from typing import List
import torch
import numpy as np

from .base_dataset import BaseDataset

class TimeSeriesForecastingDataset(BaseDataset):
    """
    A dataset class for time series forecasting problems, handling the loading, parsing, and partitioning
    of time series data into training, validation, and testing sets based on provided ratios.
    
    This class supports configurations where sequences may or may not overlap, accommodating scenarios
    where time series data is drawn from continuous periods or distinct episodes, affecting how
    the data is split into batches for model training or evaluation.
    
    Attributes:
        data_file_path (str): Path to the file containing the time series data.
        description_file_path (str): Path to the JSON file containing the description of the dataset.
        data (np.ndarray): The loaded time series data array, split according to the specified mode.
        description (dict): Metadata about the dataset, such as shape and other properties.
    """

    def __init__(self, dataset_name: str, train_val_test_ratio: List[float], mode: str, input_len: int, output_len: int, overlap: bool = True) -> None:
        """
        Initializes the TimeSeriesForecastingDataset by setting up paths, loading data, and 
        preparing it according to the specified configurations.

        Args:
            dataset_name (str): The name of the dataset.
            train_val_test_ratio (List[float]): Ratios for splitting the dataset into train, validation, and test sets.
                Each value should be a float between 0 and 1, and their sum should ideally be 1.
            mode (str): The operation mode of the dataset. Valid values are 'train', 'valid', or 'test'.
            input_len (int): The length of the input sequence (number of historical points).
            output_len (int): The length of the output sequence (number of future points to predict).
            overlap (bool): Flag to determine if training/validation/test splits should overlap.
                Defaults to True. Set to False for strictly non-overlapping periods.

        Raises:
            AssertionError: If `mode` is not one of ['train', 'valid', 'test'].
        """
        assert mode in ['train', 'valid', 'test'], f"Invalid mode: {mode}. Must be one of ['train', 'valid', 'test']."
        super().__init__(dataset_name, train_val_test_ratio, mode, input_len, output_len, overlap)

        self.data_file_path = f'datasets/{dataset_name}/data.dat'
        self.description_file_path = f'datasets/{dataset_name}/desc.json'
        self.description = self._load_description()
        self.T_long = 288
        self.data = self._load_data()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def _load_description(self) -> dict:
        """
        Loads the description of the dataset from a JSON file.

        Returns:
            dict: A dictionary containing metadata about the dataset, such as its shape and other properties.

        Raises:
            FileNotFoundError: If the description file is not found.
            json.JSONDecodeError: If there is an error decoding the JSON data.
        """
        try:
            with open(self.description_file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Description file not found: {self.description_file_path}') from e
        except json.JSONDecodeError as e:
            raise ValueError(f'Error decoding JSON file: {self.description_file_path}') from e


    def _load_data(self) -> np.ndarray:
        """
        Loads time series data from a file, splits it according to the selected mode, and sets the index range for each mode accordingly.
        
        Returns:
            np.ndarray: The index array corresponding to the specified mode (train, validation, or test).
        
        Raises:
            ValueError: If there is an issue with loading the data file or if the data shape is not as expected.
        """
        try:
            data = np.memmap(self.data_file_path, dtype='float32', mode='r', shape=tuple(self.description['shape']))
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f'Error loading data file: {self.data_file_path}') from e
        
        total_len = len(data)
        train_len = int(total_len * self.train_val_test_ratio[0])
        valid_len = int(total_len * self.train_val_test_ratio[1])
        self.train_data = data[:train_len + self.output_len].copy()
        self.valid_data = data[train_len - self.T_long + 1: train_len + valid_len + self.output_len].copy()
        self.test_data = data[train_len + valid_len - self.T_long + 1:].copy()
        if self.mode == 'train':
            offset = self.output_len if self.overlap else 0
            return data[:train_len + offset].copy()
        elif self.mode == 'valid':
            offset_left = self.input_len - 1 if self.overlap else 0
            offset_right = self.output_len if self.overlap else 0
            return data[train_len - offset_left: train_len + valid_len + offset_right].copy()
        else:  # self.mode == 'test'
            offset = self.input_len - 1 if self.overlap else 0
            return data[train_len + valid_len - offset:].copy()

    def __getitem__(self, index: int) -> dict:
        """
        Retrieves a sample from the dataset at the specified index, including historical inputs, 
        future targets, and long-term historical context.
    
        Args:
            index (int): The index of the sample to retrieve
    
        Returns:
            dict: A dictionary containing three key-value pairs:
                - inputs: Slice of historical input data, shape [input_len, ...]
                - target: Slice of future prediction data, shape [output_len, ...]
                - long_inputs: Slice of long-term historical data for capturing long-range dependencies, 
                               shape [T_long, ...]. May contain zero-padding in training mode when 
                               historical data is insufficient.
        """
        if self.mode == 'valid':
            data = self.valid_data
        elif self.mode == 'test':
            data = self.test_data
        else:
            data = self.train_data
        
        if self.mode == 'valid' or self.mode == 'test':
            long_history_data = data[index:index + self.T_long]
            history_data = data[index + self.T_long - self.input_len :index + self.T_long]
            future_data = data[index + self.T_long:index + self.T_long + self.output_len]
        elif self.mode == 'train':
            if index + self.input_len - self.T_long < 0:
                long_history_data = torch.zeros((self.T_long, data.shape[1], data.shape[2]), dtype=torch.float32)
            else:
                long_history_data = data[index - self.T_long + self.input_len:index + self.input_len]
            history_data = data[index:index + self.input_len]
            future_data = data[index + self.input_len:index + self.input_len + self.output_len]
        return {'inputs': history_data, 'target': future_data,'long_inputs': long_history_data}
    def __len__(self) -> int:
        """
        Calculates the total number of samples available in the dataset, adjusted for the lengths of input and output sequences.

        Returns:
            int: The number of valid samples that can be drawn from the dataset, based on the configurations of input and output lengths.
        """
        return len(self.data) - self.input_len - self.output_len + 1
