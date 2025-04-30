"""
Dataset source connectors for the SearchableDataset application.
"""
from .base import BaseConnector, DatasetInfo
from .kaggle import KaggleConnector
from .huggingface import HuggingFaceConnector
from .google_dataset import GoogleDatasetConnector

__all__ = [
    "BaseConnector", 
    "DatasetInfo",
    "KaggleConnector",
    "HuggingFaceConnector",
    "GoogleDatasetConnector"
]

# Registry of available connectors
CONNECTORS = {
    "kaggle": KaggleConnector,
    "huggingface": HuggingFaceConnector,
    "google_dataset": GoogleDatasetConnector,
}

def get_connector(name: str) -> BaseConnector:
    """
    Get a connector by name.
    
    Args:
        name: Name of the connector.
        
    Returns:
        BaseConnector: Connector instance.
        
    Raises:
        ValueError: If connector is not found.
    """
    if name not in CONNECTORS:
        raise ValueError(f"Connector '{name}' not found. Available connectors: {list(CONNECTORS.keys())}")
    
    return CONNECTORS[name]()
