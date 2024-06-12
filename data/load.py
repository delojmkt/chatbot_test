import pandas as pd

__all__ = ["load_data"]

def load_data(type:str, data_path:str = "./ChatbotData.csv") -> pd.DataFrame:
    """
    change this each of the data type
    """
    return pd.read_csv(data_path)