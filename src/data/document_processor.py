"""
Financial Document Analysis - Data Processing Module

This module handles loading, preprocessing, and structuring financial documents
for NLP analysis. It supports various document formats and provides utilities
for text extraction and cleaning.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import spacy
from bs4 import BeautifulSoup
from pdfplumber import PDF
from PyPDF2 import PdfReader
from python_docx import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialDocumentProcessor:
    """
    Main class for processing financial documents.
    
    Handles extraction of text from various document formats and provides
    utilities for cleaning and structuring financial text data.
    """
    
    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        """
        Initialize the document processor.
        
        Args:
            spacy_model: Name of the spaCy model to load for NLP processing
        """
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            logger.warning(f"Could not load {spacy_model}, using blank model")
            self.nlp = spacy.blank("en")
        
        # Financial entity patterns
        self.financial_patterns = {
            "revenue": r"\$\d+(?:\.\d+)?\s?(?:billion|million|thousand|B|M|K)",
            "dividend": r"\$\d+\.\d+\s?per\s?share",
            "percentage": r"\d+(?:\.\d+)?%",
            "currency": r"\$\d+(?:,\d{3})*(?:\.\d{2})?",
            "shares": r"\d+(?:,\d{3})*(?:\.\d+)?\s?(?:shares?|common\s+shares?)",
        }
    
    def extract_text_from_pdf(self, file_path: Union[str, Path]) -> str:
        """
        Extract text from PDF file using pdfplumber.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the PDF cannot be processed
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        try:
            with PDF.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            raise ValueError(f"Could not process PDF: {e}")
    
    def extract_text_from_docx(self, file_path: Union[str, Path]) -> str:
        """
        Extract text from DOCX file.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Extracted text content
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"DOCX file not found: {file_path}")
        
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            raise ValueError(f"Could not process DOCX: {e}")
    
    def extract_text_from_html(self, file_path: Union[str, Path]) -> str:
        """
        Extract text from HTML file.
        
        Args:
            file_path: Path to the HTML file
            
        Returns:
            Extracted text content
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"HTML file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                html_content = file.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text from HTML {file_path}: {e}")
            raise ValueError(f"Could not process HTML: {e}")
    
    def extract_text_from_file(self, file_path: Union[str, Path]) -> str:
        """
        Extract text from various file formats.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text content
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif suffix == '.docx':
            return self.extract_text_from_docx(file_path)
        elif suffix in ['.html', '.htm']:
            return self.extract_text_from_html(file_path)
        elif suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for processing.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep financial symbols
        text = re.sub(r'[^\w\s\$\%\.\,\-\(\)]', ' ', text)
        
        # Normalize currency symbols
        text = re.sub(r'USD|usd', '$', text)
        
        return text.strip()
    
    def extract_financial_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract financial entities using regex patterns.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary mapping entity types to lists of found entities
        """
        entities = {}
        
        for entity_type, pattern in self.financial_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities[entity_type] = matches
        
        return entities
    
    def extract_named_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract named entities using spaCy NER.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of (entity_text, entity_label) tuples
        """
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    
    def process_document(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a financial document and extract all relevant information.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Dictionary containing processed document data
        """
        logger.info(f"Processing document: {file_path}")
        
        # Extract text
        raw_text = self.extract_text_from_file(file_path)
        cleaned_text = self.clean_text(raw_text)
        
        # Extract entities
        financial_entities = self.extract_financial_entities(cleaned_text)
        named_entities = self.extract_named_entities(cleaned_text)
        
        # Count entity frequencies
        entity_counts = {}
        for entity_text, entity_label in named_entities:
            key = f"{entity_text} ({entity_label})"
            entity_counts[key] = entity_counts.get(key, 0) + 1
        
        return {
            "file_path": str(file_path),
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "financial_entities": financial_entities,
            "named_entities": named_entities,
            "entity_counts": entity_counts,
            "text_length": len(cleaned_text),
            "word_count": len(cleaned_text.split()),
        }
    
    def process_document_from_text(self, text: str) -> Dict[str, Any]:
        """
        Process financial document text directly.
        
        Args:
            text: Document text to process
            
        Returns:
            Dictionary containing processed document data
        """
        logger.info("Processing document from text")
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Extract entities
        financial_entities = self.extract_financial_entities(cleaned_text)
        named_entities = self.extract_named_entities(cleaned_text)
        
        # Count entity frequencies
        entity_counts = {}
        for entity_text, entity_label in named_entities:
            key = f"{entity_text} ({entity_label})"
            entity_counts[key] = entity_counts.get(key, 0) + 1
        
        return {
            "raw_text": text,
            "cleaned_text": cleaned_text,
            "financial_entities": financial_entities,
            "named_entities": named_entities,
            "entity_counts": entity_counts,
            "text_length": len(cleaned_text),
            "word_count": len(cleaned_text.split()),
        }


def create_sample_financial_documents() -> List[Dict[str, Any]]:
    """
    Create sample financial documents for testing and demonstration.
    
    Returns:
        List of sample document dictionaries
    """
    sample_docs = [
        {
            "title": "Apple Inc. Q4 2023 Earnings Report",
            "content": """
            Apple Inc. reported a quarterly revenue of $123.9 billion, a 10% increase year-over-year.
            The company also declared a dividend of $0.22 per share, which will be paid on April 1, 2024.
            CEO Tim Cook announced a 5% growth in their services segment, contributing $19 billion to the revenue.
            In addition, Apple's cash reserves were reported to be approximately $75 billion.
            The company's stock price increased by 3.2% following the earnings announcement.
            """,
            "metadata": {
                "company": "Apple Inc.",
                "quarter": "Q4 2023",
                "document_type": "earnings_report",
                "date": "2023-10-26"
            }
        },
        {
            "title": "Microsoft Corporation Annual Report 2023",
            "content": """
            Microsoft Corporation reported annual revenue of $211.9 billion for fiscal year 2023.
            The company's cloud services segment, Azure, grew by 28% year-over-year.
            Microsoft declared a quarterly dividend of $0.75 per share.
            The company's market capitalization reached $2.8 trillion.
            Revenue from Office 365 increased by 12% compared to the previous year.
            """,
            "metadata": {
                "company": "Microsoft Corporation",
                "year": "2023",
                "document_type": "annual_report",
                "date": "2023-07-28"
            }
        },
        {
            "title": "Tesla Inc. Q3 2023 Financial Results",
            "content": """
            Tesla Inc. reported quarterly revenue of $23.4 billion in Q3 2023.
            The company delivered 435,059 vehicles during the quarter.
            Tesla's gross margin improved to 19.3% from 18.2% in the previous quarter.
            The company's energy storage deployments increased by 90% year-over-year.
            Tesla's stock price closed at $242.68 per share.
            """,
            "metadata": {
                "company": "Tesla Inc.",
                "quarter": "Q3 2023",
                "document_type": "earnings_report",
                "date": "2023-10-18"
            }
        }
    ]
    
    return sample_docs


if __name__ == "__main__":
    # Example usage
    processor = FinancialDocumentProcessor()
    
    # Process sample documents
    sample_docs = create_sample_financial_documents()
    
    for doc in sample_docs:
        print(f"\n=== {doc['title']} ===")
        result = processor.process_document_from_text(doc['content'])
        
        print(f"Text length: {result['text_length']} characters")
        print(f"Word count: {result['word_count']}")
        
        print("\nFinancial Entities:")
        for entity_type, entities in result['financial_entities'].items():
            if entities:
                print(f"  {entity_type}: {entities}")
        
        print("\nNamed Entities:")
        for entity_text, entity_label in result['named_entities'][:5]:  # Show first 5
            print(f"  {entity_text} ({entity_label})")
        
        print("\nEntity Counts:")
        for entity, count in list(result['entity_counts'].items())[:5]:  # Show first 5
            print(f"  {entity}: {count}")
