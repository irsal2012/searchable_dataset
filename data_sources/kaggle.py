"""
Kaggle dataset connector.
"""
import os
from typing import List, Optional, Dict, Any
import kaggle
from .base import BaseConnector, DatasetInfo
from utils import config

class KaggleConnector(BaseConnector):
    """Connector for Kaggle datasets."""
    
    def __init__(self):
        """Initialize the Kaggle connector."""
        super().__init__("kaggle")
        
        # Set Kaggle credentials from config
        kaggle_config = config.get_kaggle_config()
        os.environ["KAGGLE_USERNAME"] = kaggle_config["username"]
        os.environ["KAGGLE_KEY"] = kaggle_config["key"]
    
    def search(self, query: str, limit: int = 10) -> List[DatasetInfo]:
        """
        Search for datasets on Kaggle.
        
        Args:
            query: Search query.
            limit: Maximum number of results.
            
        Returns:
            List[DatasetInfo]: List of dataset information.
        """
        try:
            # Search for datasets
            datasets = kaggle.api.dataset_list(search=query, max_size=limit)
            
            # Convert to DatasetInfo objects
            results = []
            for dataset in datasets:
                dataset_info = self._convert_to_dataset_info(dataset)
                results.append(dataset_info)
            
            return results
        except Exception as e:
            self.logger.error(f"Error searching Kaggle datasets: {e}")
            return []
    
    def get_dataset(self, dataset_id: str) -> Optional[DatasetInfo]:
        """
        Get dataset information by ID.
        
        Args:
            dataset_id: Dataset ID in the format "username/dataset-name".
            
        Returns:
            Optional[DatasetInfo]: Dataset information or None if not found.
        """
        try:
            # Get dataset information
            dataset = kaggle.api.dataset_view(dataset_id)
            
            # Convert to DatasetInfo
            return self._convert_to_dataset_info(dataset)
        except Exception as e:
            self.logger.error(f"Error getting Kaggle dataset '{dataset_id}': {e}")
            return None
    
    def _convert_to_dataset_info(self, dataset: Any) -> DatasetInfo:
        """
        Convert Kaggle dataset to DatasetInfo.
        
        Args:
            dataset: Kaggle dataset object.
            
        Returns:
            DatasetInfo: Dataset information.
        """
        # Extract metadata
        metadata: Dict[str, Any] = {}
        for key, value in dataset.__dict__.items():
            if key not in ["id", "name", "description", "url", "size", "license", "tags"]:
                metadata[key] = value
        
        # Get size safely
        size = None
        try:
            size = dataset.size
        except AttributeError:
            self.logger.warning(f"Dataset {dataset.ref} does not have a size attribute")
        
        # Create DatasetInfo
        return DatasetInfo(
            id=dataset.ref,
            name=dataset.title,
            description=dataset.subtitle or "",
            source="Kaggle",
            url=f"https://www.kaggle.com/datasets/{dataset.ref}",
            size=self._format_size(size),
            format=None,  # Kaggle API doesn't provide format information
            license=dataset.licenseName if hasattr(dataset, "licenseName") else "Unknown",
            tags=[tag.name for tag in dataset.tags] if hasattr(dataset, "tags") else [],
            metadata=metadata,
        )
    
    def _format_size(self, size_bytes: int) -> str:
        """
        Format size in bytes to human-readable format.
        
        Args:
            size_bytes: Size in bytes.
            
        Returns:
            str: Formatted size.
        """
        if size_bytes is None:
            return "Unknown"
        
        # Convert to appropriate unit
        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(size_bytes)
        unit_index = 0
        
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        
        return f"{size:.2f} {units[unit_index]}"
