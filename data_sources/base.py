"""
Base connector class for dataset sources.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from utils import logger, cache

class DatasetInfo:
    """Class representing dataset information."""
    
    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        source: str,
        url: Optional[str] = None,
        size: Optional[str] = None,
        format: Optional[str] = None,
        license: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize dataset information.
        
        Args:
            id: Unique identifier for the dataset.
            name: Name of the dataset.
            description: Description of the dataset.
            source: Source of the dataset (e.g., "Kaggle", "Hugging Face").
            url: URL to the dataset.
            size: Size of the dataset.
            format: Format of the dataset.
            license: License of the dataset.
            tags: Tags associated with the dataset.
            metadata: Additional metadata.
        """
        self.id = id
        self.name = name
        self.description = description
        self.source = source
        self.url = url
        self.size = size
        self.format = format
        self.license = license
        self.tags = tags or []
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "url": self.url,
            "size": self.size,
            "format": self.format,
            "license": self.license,
            "tags": self.tags,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetInfo":
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation.
            
        Returns:
            DatasetInfo: Dataset information.
        """
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            source=data["source"],
            url=data.get("url"),
            size=data.get("size"),
            format=data.get("format"),
            license=data.get("license"),
            tags=data.get("tags"),
            metadata=data.get("metadata"),
        )


class BaseConnector(ABC):
    """Base class for dataset source connectors."""
    
    def __init__(self, name: str):
        """
        Initialize the connector.
        
        Args:
            name: Name of the connector.
        """
        self.name = name
        self.logger = logger.setup_logger(f"connector.{name}")
    
    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[DatasetInfo]:
        """
        Search for datasets.
        
        Args:
            query: Search query.
            limit: Maximum number of results.
            
        Returns:
            List[DatasetInfo]: List of dataset information.
        """
        pass
    
    @abstractmethod
    def get_dataset(self, dataset_id: str) -> Optional[DatasetInfo]:
        """
        Get dataset information by ID.
        
        Args:
            dataset_id: Dataset ID.
            
        Returns:
            Optional[DatasetInfo]: Dataset information or None if not found.
        """
        pass
    
    @cache.cached
    def search_cached(self, query: str, limit: int = 10) -> List[DatasetInfo]:
        """
        Search for datasets with caching.
        
        Args:
            query: Search query.
            limit: Maximum number of results.
            
        Returns:
            List[DatasetInfo]: List of dataset information.
        """
        self.logger.info(f"Searching for '{query}' (limit={limit})")
        return self.search(query, limit)
    
    @cache.cached
    def get_dataset_cached(self, dataset_id: str) -> Optional[DatasetInfo]:
        """
        Get dataset information by ID with caching.
        
        Args:
            dataset_id: Dataset ID.
            
        Returns:
            Optional[DatasetInfo]: Dataset information or None if not found.
        """
        self.logger.info(f"Getting dataset '{dataset_id}'")
        return self.get_dataset(dataset_id)
