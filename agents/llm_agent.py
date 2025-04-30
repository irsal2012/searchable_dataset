"""
LLM agent for dataset search and analysis.
"""
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

from .prompts import PromptTemplates
from .processors import ResponseProcessor
from data_sources import get_connector, DatasetInfo
from utils import config, cache
from utils.logger import setup_logger

class LLMAgent:
    """LLM agent for dataset search and analysis."""
    
    def __init__(self):
        """Initialize the LLM agent."""
        self.logger = setup_logger("llm_agent")
        
        # Set up OpenAI client
        openai.api_key = config.OPENAI_API_KEY
        
        # Set up LangChain models
        self.llm_config = config.get_llm_config()
        self.chat_model = ChatOpenAI(
            model_name=self.llm_config["model"],
            temperature=self.llm_config["temperature"],
            openai_api_key=self.llm_config["api_key"],
        )
    
    def search_datasets(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Search for datasets based on a user query.
        
        Args:
            query: User query.
            context: Additional context.
            
        Returns:
            Dict[str, Any]: Search results.
        """
        self.logger.info(f"Searching datasets for query: {query}")
        
        # Generate search terms using LLM
        search_terms, data_sources, explanation = self._generate_search_terms(query, context)
        
        # If no data sources specified, use all available
        if not data_sources:
            from data_sources import CONNECTORS
            data_sources = list(CONNECTORS.keys())
        
        # Search datasets from multiple sources in parallel
        all_datasets = self._search_multiple_sources(search_terms, data_sources)
        
        # Analyze datasets using LLM
        analysis = self._analyze_datasets(query, all_datasets)
        
        return {
            "query": query,
            "search_terms": search_terms,
            "data_sources": data_sources,
            "explanation": explanation,
            "datasets": all_datasets,
            "analysis": analysis,
        }
    
    def get_dataset_recommendation(self, query: str, datasets: List[Dict[str, Any]]) -> str:
        """
        Get a recommendation for the best dataset based on a user query.
        
        Args:
            query: User query.
            datasets: List of datasets.
            
        Returns:
            str: Recommendation.
        """
        self.logger.info(f"Getting dataset recommendation for query: {query}")
        
        # Generate prompt
        prompt = PromptTemplates.dataset_recommendation_prompt(query, datasets)
        
        # Get recommendation from LLM
        response = self._call_llm(prompt)
        
        # Process response
        recommendation = ResponseProcessor.process_dataset_recommendation(response)
        
        return recommendation
    
    def _generate_search_terms(self, query: str, context: Dict[str, Any] = None) -> Tuple[List[str], List[str], str]:
        """
        Generate search terms for a user query.
        
        Args:
            query: User query.
            context: Additional context.
            
        Returns:
            Tuple[List[str], List[str], str]: Search terms, data sources, and explanation.
        """
        # Generate prompt
        prompt = PromptTemplates.dataset_search_prompt(query, context)
        
        # Get response from LLM
        response = self._call_llm(prompt)
        
        # Process response
        search_terms, data_sources, explanation = ResponseProcessor.process_search_terms(response)
        
        self.logger.info(f"Generated search terms: {search_terms}")
        self.logger.info(f"Recommended data sources: {data_sources}")
        
        return search_terms, data_sources, explanation
    
    def _search_multiple_sources(self, search_terms: List[str], data_sources: List[str]) -> List[Dict[str, Any]]:
        """
        Search for datasets from multiple sources.
        
        Args:
            search_terms: List of search terms.
            data_sources: List of data sources.
            
        Returns:
            List[Dict[str, Any]]: List of datasets.
        """
        all_datasets = []
        
        # Create a thread pool
        with ThreadPoolExecutor(max_workers=len(data_sources)) as executor:
            # Submit search tasks
            future_to_source = {}
            for source in data_sources:
                future = executor.submit(self._search_source, source, search_terms)
                future_to_source[future] = source
            
            # Process results as they complete
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    datasets = future.result()
                    self.logger.info(f"Found {len(datasets)} datasets from {source}")
                    all_datasets.extend(datasets)
                except Exception as e:
                    self.logger.error(f"Error searching {source}: {e}")
        
        return all_datasets
    
    def _search_source(self, source: str, search_terms: List[str]) -> List[Dict[str, Any]]:
        """
        Search for datasets from a specific source.
        
        Args:
            source: Data source.
            search_terms: List of search terms.
            
        Returns:
            List[Dict[str, Any]]: List of datasets.
        """
        try:
            # Get connector
            connector = get_connector(source)
            
            # Search for each term
            all_results = []
            for term in search_terms:
                results = connector.search_cached(term)
                all_results.extend(results)
            
            # Remove duplicates
            unique_results = {}
            for dataset in all_results:
                if dataset.id not in unique_results:
                    unique_results[dataset.id] = dataset
            
            # Convert to dictionaries
            return [dataset.to_dict() for dataset in unique_results.values()]
        except Exception as e:
            self.logger.error(f"Error searching {source}: {e}")
            return []
    
    def _analyze_datasets(self, query: str, datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze datasets using LLM.
        
        Args:
            query: User query.
            datasets: List of datasets.
            
        Returns:
            Dict[str, Any]: Analysis results.
        """
        # Limit the number of datasets to analyze
        max_datasets = 10
        if len(datasets) > max_datasets:
            self.logger.info(f"Limiting analysis to {max_datasets} datasets")
            datasets = datasets[:max_datasets]
        
        # Generate prompt
        prompt = PromptTemplates.dataset_analysis_prompt(query, datasets)
        
        # Get response from LLM
        response = self._call_llm(prompt)
        
        # Process response
        analysis = ResponseProcessor.process_dataset_analysis(response)
        
        return analysis
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM with a prompt.
        
        Args:
            prompt: Prompt for the LLM.
            
        Returns:
            str: LLM response.
        """
        try:
            # Call the LLM
            response = self.chat_model([HumanMessage(content=prompt)])
            
            # Extract content
            content = response.content
            
            return content
        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            return ""
