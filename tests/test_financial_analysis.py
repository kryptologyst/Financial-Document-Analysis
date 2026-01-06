"""
Unit tests for Financial Document Analysis system.

Tests cover document processing, NLP models, evaluation metrics,
and core functionality.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import numpy as np

# Import modules to test
from src.data.document_processor import FinancialDocumentProcessor, create_sample_financial_documents
from src.models.nlp_models import (
    FinancialSentimentAnalyzer,
    FinancialNER,
    FinancialDocumentClassifier,
    FinancialTextSummarizer,
    FinancialInsightExtractor
)
from src.models.evaluation import NLPEvaluator, FinancialMetricsEvaluator


class TestFinancialDocumentProcessor:
    """Test cases for FinancialDocumentProcessor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = FinancialDocumentProcessor()
        self.sample_text = """
        Apple Inc. reported a quarterly revenue of $123.9 billion, a 10% increase year-over-year.
        The company also declared a dividend of $0.22 per share, which will be paid on April 1, 2024.
        CEO Tim Cook announced a 5% growth in their services segment.
        """
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        dirty_text = "Apple Inc.   reported   a quarterly revenue of $123.9 billion!!!"
        cleaned = self.processor.clean_text(dirty_text)
        
        assert "  " not in cleaned  # No double spaces
        assert "!!!" not in cleaned  # Special characters removed
        assert "$123.9 billion" in cleaned  # Financial symbols preserved
    
    def test_extract_financial_entities(self):
        """Test financial entity extraction."""
        entities = self.processor.extract_financial_entities(self.sample_text)
        
        assert "revenue" in entities
        assert "dividend" in entities
        assert "percentage" in entities
        assert "$123.9 billion" in entities["revenue"]
        assert "$0.22 per share" in entities["dividend"]
        assert "10%" in entities["percentage"]
    
    def test_extract_named_entities(self):
        """Test named entity recognition."""
        entities = self.processor.extract_named_entities(self.sample_text)
        
        # Should extract company names and other entities
        entity_texts = [ent[0] for ent in entities]
        assert "Apple Inc." in entity_texts
        assert "Tim Cook" in entity_texts
    
    def test_create_sample_documents(self):
        """Test sample document creation."""
        sample_docs = create_sample_financial_documents()
        
        assert len(sample_docs) > 0
        assert all("title" in doc for doc in sample_docs)
        assert all("content" in doc for doc in sample_docs)
        assert all("metadata" in doc for doc in sample_docs)


class TestFinancialSentimentAnalyzer:
    """Test cases for FinancialSentimentAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock the model loading to avoid downloading during tests
        with patch('src.models.nlp_models.AutoTokenizer') as mock_tokenizer, \
             patch('src.models.nlp_models.AutoModelForSequenceClassification') as mock_model, \
             patch('src.models.nlp_models.pipeline') as mock_pipeline:
            
            mock_tokenizer.from_pretrained.return_value = Mock()
            mock_model.from_pretrained.return_value = Mock()
            mock_pipeline.return_value = Mock()
            
            self.analyzer = FinancialSentimentAnalyzer()
    
    def test_analyze_sentiment(self):
        """Test sentiment analysis."""
        positive_text = "Apple reported strong quarterly growth and exceeded expectations."
        
        # Mock the pipeline result
        mock_result = [{'label': 'LABEL_2', 'score': 0.85}]
        self.analyzer.sentiment_pipeline.return_value = mock_result
        
        result = self.analyzer.analyze_sentiment(positive_text)
        
        assert result['sentiment'] == 'positive'
        assert result['confidence'] == 0.85
        assert 'text' in result
    
    def test_analyze_batch(self):
        """Test batch sentiment analysis."""
        texts = [
            "Revenue increased significantly",
            "Profit declined this quarter",
            "Market outlook remains stable"
        ]
        
        # Mock batch results
        mock_results = [
            [{'label': 'LABEL_2', 'score': 0.8}],
            [{'label': 'LABEL_0', 'score': 0.7}],
            [{'label': 'LABEL_1', 'score': 0.6}]
        ]
        self.analyzer.sentiment_pipeline.side_effect = mock_results
        
        results = self.analyzer.analyze_batch(texts)
        
        assert len(results) == 3
        assert results[0]['sentiment'] == 'positive'
        assert results[1]['sentiment'] == 'negative'
        assert results[2]['sentiment'] == 'neutral'


class TestFinancialNER:
    """Test cases for FinancialNER."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ner = FinancialNER()
        self.sample_text = """
        Apple Inc. reported quarterly revenue of $123.9 billion.
        Microsoft Corporation declared a dividend of $0.75 per share.
        Tesla's stock price increased by 15% to $250.00.
        """
    
    def test_extract_financial_entities(self):
        """Test financial entity extraction."""
        entities = self.ner.extract_financial_entities(self.sample_text)
        
        assert "REVENUE" in entities
        assert "DIVIDEND" in entities
        assert "PERCENTAGE" in entities
        assert "$123.9 billion" in entities["REVENUE"]
        assert "$0.75 per share" in entities["DIVIDEND"]
        assert "15%" in entities["PERCENTAGE"]
    
    def test_extract_companies(self):
        """Test company name extraction."""
        companies = self.ner.extract_companies(self.sample_text)
        
        assert "Apple Inc." in companies
        assert "Microsoft Corporation" in companies
        assert "Tesla" in companies


class TestFinancialDocumentClassifier:
    """Test cases for FinancialDocumentClassifier."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = FinancialDocumentClassifier()
    
    def test_classify_earnings_report(self):
        """Test classification of earnings report."""
        earnings_text = """
        Apple Inc. reported quarterly earnings of $1.52 per share.
        Revenue increased 10% year-over-year to $123.9 billion.
        The company provided guidance for the next quarter.
        """
        
        result = self.classifier.classify_document(earnings_text)
        
        assert result['predicted_type'] == 'earnings_report'
        assert result['confidence'] > 0
    
    def test_classify_annual_report(self):
        """Test classification of annual report."""
        annual_text = """
        This annual report covers the fiscal year 2023.
        The financial statements have been audited by our auditors.
        The balance sheet shows strong cash position.
        """
        
        result = self.classifier.classify_document(annual_text)
        
        assert result['predicted_type'] == 'annual_report'
        assert result['confidence'] > 0


class TestFinancialTextSummarizer:
    """Test cases for FinancialTextSummarizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.summarizer = FinancialTextSummarizer()
        self.long_text = """
        Apple Inc. reported a quarterly revenue of $123.9 billion, a 10% increase year-over-year.
        The company also declared a dividend of $0.22 per share, which will be paid on April 1, 2024.
        CEO Tim Cook announced a 5% growth in their services segment, contributing $19 billion to the revenue.
        In addition, Apple's cash reserves were reported to be approximately $75 billion.
        The company's stock price increased by 3.2% following the earnings announcement.
        Apple's iPhone sales grew 8% compared to the previous quarter.
        The services business showed strong performance with 15% growth.
        """
    
    def test_extractive_summarize(self):
        """Test extractive summarization."""
        summary = self.summarizer.extractive_summarize(self.long_text, num_sentences=3)
        
        assert len(summary.split('.')) <= 4  # Should be around 3 sentences
        assert "revenue" in summary.lower()  # Should contain key financial terms
        assert len(summary) < len(self.long_text)  # Should be shorter than original


class TestFinancialInsightExtractor:
    """Test cases for FinancialInsightExtractor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = FinancialInsightExtractor()
        self.sample_text = """
        Apple Inc. reported quarterly revenue of $123.9 billion.
        Net profit was $25.8 billion for the quarter.
        EPS was $1.52 per share.
        The company declared a dividend of $0.22 per share.
        Revenue growth was 10% year-over-year.
        Gross margin improved to 42.5%.
        """
    
    def test_extract_metrics(self):
        """Test financial metric extraction."""
        metrics = self.extractor.extract_metrics(self.sample_text)
        
        assert "revenue" in metrics
        assert "profit" in metrics
        assert "eps" in metrics
        assert "dividend" in metrics
        assert "growth" in metrics
        assert "margin" in metrics
    
    def test_extract_trends(self):
        """Test trend extraction."""
        trend_text = """
        Revenue increased significantly this quarter.
        Profit margins declined due to higher costs.
        Market share grew in all segments.
        """
        
        trends = self.extractor.extract_trends(trend_text)
        
        assert len(trends) > 0
        assert any("increased" in trend.lower() for trend in trends)
        assert any("declined" in trend.lower() for trend in trends)


class TestNLPEvaluator:
    """Test cases for NLPEvaluator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = NLPEvaluator()
    
    def test_evaluate_sentiment_analysis(self):
        """Test sentiment analysis evaluation."""
        predictions = ['positive', 'negative', 'neutral', 'positive']
        ground_truth = ['positive', 'negative', 'positive', 'positive']
        
        metrics = self.evaluator.evaluate_sentiment_analysis(predictions, ground_truth)
        
        assert 'accuracy' in metrics
        assert 'precision_macro' in metrics
        assert 'recall_macro' in metrics
        assert 'f1_macro' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_evaluate_ner(self):
        """Test NER evaluation."""
        predictions = [
            [('Apple Inc.', 'ORG'), ('$123.9 billion', 'MONEY')],
            [('Microsoft', 'ORG'), ('$0.75', 'MONEY')]
        ]
        ground_truth = [
            [('Apple Inc.', 'ORG'), ('$123.9 billion', 'MONEY')],
            [('Microsoft Corporation', 'ORG'), ('$0.75', 'MONEY')]
        ]
        
        metrics = self.evaluator.evaluate_ner(predictions, ground_truth)
        
        assert 'exact_match_accuracy' in metrics
        assert 'total_entities' in metrics
        assert 0 <= metrics['exact_match_accuracy'] <= 1
    
    def test_evaluate_document_classification(self):
        """Test document classification evaluation."""
        predictions = ['earnings_report', 'annual_report', 'press_release']
        ground_truth = ['earnings_report', 'annual_report', 'earnings_report']
        
        metrics = self.evaluator.evaluate_document_classification(predictions, ground_truth)
        
        assert 'accuracy' in metrics
        assert 'classification_report' in metrics
        assert 'confusion_matrix' in metrics
        assert 0 <= metrics['accuracy'] <= 1


class TestFinancialMetricsEvaluator:
    """Test cases for FinancialMetricsEvaluator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = FinancialMetricsEvaluator()
    
    def test_evaluate_metric_extraction_accuracy(self):
        """Test metric extraction accuracy evaluation."""
        predictions = [
            {'revenue': '$123.9 billion', 'profit': '$25.8 billion'},
            {'revenue': '$98.5 billion', 'profit': '$20.1 billion'}
        ]
        ground_truth = [
            {'revenue': '$123.9 billion', 'profit': '$25.8 billion'},
            {'revenue': '$98.5 billion', 'profit': '$20.1 billion'}
        ]
        
        metrics = self.evaluator.evaluate_metric_extraction_accuracy(predictions, ground_truth)
        
        assert 'revenue_extraction_accuracy' in metrics
        assert 'profit_extraction_accuracy' in metrics
        assert metrics['revenue_extraction_accuracy'] == 1.0
        assert metrics['profit_extraction_accuracy'] == 1.0
    
    def test_evaluate_trend_detection(self):
        """Test trend detection evaluation."""
        predictions = [
            ['Revenue increased', 'Profit grew'],
            ['Market share declined', 'Costs rose']
        ]
        ground_truth = [
            ['Revenue increased', 'Profit grew'],
            ['Market share declined', 'Costs increased']
        ]
        
        metrics = self.evaluator.evaluate_trend_detection(predictions, ground_truth)
        
        assert 'trend_precision' in metrics
        assert 'trend_recall' in metrics
        assert 'trend_f1' in metrics
        assert 0 <= metrics['trend_precision'] <= 1


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_document_processing(self):
        """Test complete document processing pipeline."""
        processor = FinancialDocumentProcessor()
        sample_docs = create_sample_financial_documents()
        
        # Process first sample document
        doc = sample_docs[0]
        result = processor.process_document_from_text(doc['content'])
        
        assert 'cleaned_text' in result
        assert 'financial_entities' in result
        assert 'named_entities' in result
        assert 'text_length' in result
        assert 'word_count' in result
        
        # Verify financial entities were extracted
        assert len(result['financial_entities']) > 0
        assert any(len(entities) > 0 for entities in result['financial_entities'].values())
    
    def test_model_compatibility(self):
        """Test that all models work together."""
        # This test ensures models can be instantiated without errors
        try:
            processor = FinancialDocumentProcessor()
            ner = FinancialNER()
            classifier = FinancialDocumentClassifier()
            summarizer = FinancialTextSummarizer()
            extractor = FinancialInsightExtractor()
            
            # All models should be instantiable
            assert processor is not None
            assert ner is not None
            assert classifier is not None
            assert summarizer is not None
            assert extractor is not None
            
        except Exception as e:
            pytest.fail(f"Model compatibility test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
