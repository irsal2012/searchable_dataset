"""
Hugging Face dataset connector.
"""
from typing import List, Optional, Dict, Any
import requests
from huggingface_hub import HfApi
from .base import BaseConnector, DatasetInfo
from utils import config

class HuggingFaceConnector(BaseConnector):
    """Connector for Hugging Face datasets."""
    
    def __init__(self):
        """Initialize the Hugging Face connector."""
        super().__init__("huggingface")
        
        # Set up API
        hf_config = config.get_huggingface_config()
        self.api_key = hf_config["api_key"]
        self.api = HfApi(token=self.api_key)
        self.base_url = "https://huggingface.co/api/datasets"
    
    def search(self, query: str, limit: int = 10) -> List[DatasetInfo]:
        """
        Search for datasets on Hugging Face.
        
        Args:
            query: Search query.
            limit: Maximum number of results.
            
        Returns:
            List[DatasetInfo]: List of dataset information.
        """
        try:
            # Search for datasets
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            params = {"search": query, "limit": limit}
            response = requests.get(self.base_url, headers=headers, params=params)
            response.raise_for_status()
            
            # Parse response
            datasets = response.json()
            
            # Convert to DatasetInfo objects
            results = []
            for dataset in datasets:
                dataset_info = self._convert_to_dataset_info(dataset)
                results.append(dataset_info)
            
            return results
        except Exception as e:
            self.logger.error(f"Error searching Hugging Face datasets: {e}")
            return []
    
    def get_dataset(self, dataset_id: str) -> Optional[DatasetInfo]:
        """
        Get dataset information by ID.
        
        Args:
            dataset_id: Dataset ID.
            
        Returns:
            Optional[DatasetInfo]: Dataset information or None if not found.
        """
        try:
            # Get dataset information
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            response = requests.get(f"{self.base_url}/{dataset_id}", headers=headers)
            response.raise_for_status()
            
            # Parse response
            dataset = response.json()
            
            # Convert to DatasetInfo
            return self._convert_to_dataset_info(dataset)
        except Exception as e:
            self.logger.error(f"Error getting Hugging Face dataset '{dataset_id}': {e}")
            return None
    
    def _convert_to_dataset_info(self, dataset: Dict[str, Any]) -> DatasetInfo:
        """
        Convert Hugging Face dataset to DatasetInfo.
        
        Args:
            dataset: Hugging Face dataset dictionary.
            
        Returns:
            DatasetInfo: Dataset information.
        """
        # Extract metadata
        metadata: Dict[str, Any] = {}
        for key, value in dataset.items():
            if key not in ["id", "name", "description", "url", "size", "license", "tags"]:
                metadata[key] = value
        
        # Create DatasetInfo
        return DatasetInfo(
            id=dataset["id"],
            name=dataset.get("name", dataset["id"]),
            description=dataset.get("description", ""),
            source="Hugging Face",
            url=f"https://huggingface.co/datasets/{dataset['id']}",
            size=dataset.get("size_categories", ["Unknown"])[0],
            format=None,  # Hugging Face API doesn't provide format information
            license=dataset.get("license", "Unknown"),
            tags=dataset.get("tags", []),
            metadata=metadata,
        )
