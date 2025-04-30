"""
Google Dataset Search connector.
"""
import re
import requests
from typing import List, Optional, Dict, Any
from bs4 import BeautifulSoup
from .base import BaseConnector, DatasetInfo
from utils import logger

class GoogleDatasetConnector(BaseConnector):
    """Connector for Google Dataset Search."""
    
    def __init__(self):
        """Initialize the Google Dataset Search connector."""
        super().__init__("google_dataset")
        self.base_url = "https://datasetsearch.research.google.com/search"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        }
    
    def search(self, query: str, limit: int = 10) -> List[DatasetInfo]:
        """
        Search for datasets on Google Dataset Search.
        
        Args:
            query: Search query.
            limit: Maximum number of results.
            
        Returns:
            List[DatasetInfo]: List of dataset information.
        """
        try:
            # Prepare search URL
            params = {"query": query}
            response = requests.get(self.base_url, headers=self.headers, params=params)
            response.raise_for_status()
            
            # Parse HTML response
            soup = BeautifulSoup(response.text, "lxml")
            
            # Extract dataset information
            dataset_elements = soup.select(".dataset-card")
            
            # Limit results
            dataset_elements = dataset_elements[:limit]
            
            # Convert to DatasetInfo objects
            results = []
            for element in dataset_elements:
                try:
                    dataset_info = self._extract_dataset_info(element)
                    if dataset_info:
                        results.append(dataset_info)
                except Exception as e:
                    self.logger.error(f"Error extracting dataset info: {e}")
            
            return results
        except Exception as e:
            self.logger.error(f"Error searching Google Dataset Search: {e}")
            return []
    
    def get_dataset(self, dataset_id: str) -> Optional[DatasetInfo]:
        """
        Get dataset information by ID.
        
        Args:
            dataset_id: Dataset ID (URL).
            
        Returns:
            Optional[DatasetInfo]: Dataset information or None if not found.
        """
        try:
            # For Google Dataset Search, the ID is the URL
            response = requests.get(dataset_id, headers=self.headers)
            response.raise_for_status()
            
            # Parse HTML response
            soup = BeautifulSoup(response.text, "lxml")
            
            # Extract dataset information
            # This is a simplified implementation and may need to be adapted
            # based on the actual structure of the dataset page
            name = soup.select_one("h1").text.strip()
            description = soup.select_one("meta[name='description']")["content"]
            
            # Create DatasetInfo
            return DatasetInfo(
                id=dataset_id,
                name=name,
                description=description,
                source="Google Dataset Search",
                url=dataset_id,
                size=None,
                format=None,
                license=None,
                tags=[],
                metadata={},
            )
        except Exception as e:
            self.logger.error(f"Error getting Google dataset '{dataset_id}': {e}")
            return None
    
    def _extract_dataset_info(self, element: Any) -> Optional[DatasetInfo]:
        """
        Extract dataset information from HTML element.
        
        Args:
            element: BeautifulSoup element.
            
        Returns:
            Optional[DatasetInfo]: Dataset information or None if extraction fails.
        """
        try:
            # Extract basic information
            title_element = element.select_one(".dataset-title")
            if not title_element:
                return None
            
            name = title_element.text.strip()
            
            # Extract URL
            url_element = title_element.find("a")
            url = url_element["href"] if url_element else None
            
            # Extract description
            description_element = element.select_one(".dataset-description")
            description = description_element.text.strip() if description_element else ""
            
            # Extract additional information
            metadata: Dict[str, Any] = {}
            info_elements = element.select(".dataset-info-item")
            for info in info_elements:
                label_element = info.select_one(".info-label")
                value_element = info.select_one(".info-value")
                
                if label_element and value_element:
                    label = label_element.text.strip().lower().replace(" ", "_")
                    value = value_element.text.strip()
                    metadata[label] = value
            
            # Extract size
            size = metadata.get("size", None)
            
            # Extract format
            format_value = metadata.get("file_format", None)
            
            # Extract license
            license_value = metadata.get("license", None)
            
            # Extract tags
            tags = []
            tags_element = element.select_one(".dataset-tags")
            if tags_element:
                tag_elements = tags_element.select(".dataset-tag")
                tags = [tag.text.strip() for tag in tag_elements]
            
            # Generate a unique ID if URL is not available
            if not url:
                # Create a hash of the name and description
                id_value = f"google_{hash(name + description)}"
            else:
                # Use the URL as the ID
                id_value = url
            
            # Create DatasetInfo
            return DatasetInfo(
                id=id_value,
                name=name,
                description=description,
                source="Google Dataset Search",
                url=url,
                size=size,
                format=format_value,
                license=license_value,
                tags=tags,
                metadata=metadata,
            )
        except Exception as e:
            self.logger.error(f"Error extracting dataset info: {e}")
            return None
