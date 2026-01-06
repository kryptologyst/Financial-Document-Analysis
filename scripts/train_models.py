#!/usr/bin/env python3
"""
Training script for Financial Document Analysis models.

This script demonstrates how to train and evaluate the NLP models
on financial document datasets.
"""

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Import our modules
from src.data.document_processor import FinancialDocumentProcessor, create_sample_financial_documents
from src.models.nlp_models import (
    FinancialSentimentAnalyzer,
    FinancialNER,
    FinancialDocumentClassifier,
    FinancialTextSummarizer,
    FinancialInsightExtractor
)
from src.models.evaluation import NLPEvaluator, FinancialMetricsEvaluator, ModelComparisonEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


def create_sample_dataset() -> pd.DataFrame:
    """
    Create a sample dataset for training and evaluation.
    
    Returns:
        DataFrame with sample financial documents and labels
    """
    sample_docs = create_sample_financial_documents()
    
    # Create additional sample data
    additional_samples = [
        {
            "title": "Amazon Q3 2023 Earnings",
            "content": "Amazon reported net sales of $143.1 billion, up 13% year-over-year. AWS revenue grew 12% to $23.1 billion. The company's operating income increased to $11.2 billion.",
            "metadata": {"company": "Amazon", "quarter": "Q3 2023", "document_type": "earnings_report", "date": "2023-10-26"},
            "sentiment": "positive",
            "document_type": "earnings_report"
        },
        {
            "title": "Meta Platforms Q3 2023 Results",
            "content": "Meta reported revenue of $34.1 billion, up 23% year-over-year. Daily active users increased to 3.14 billion. Reality Labs revenue was $210 million.",
            "metadata": {"company": "Meta", "quarter": "Q3 2023", "document_type": "earnings_report", "date": "2023-10-25"},
            "sentiment": "positive",
            "document_type": "earnings_report"
        },
        {
            "title": "Netflix Q3 2023 Earnings Call",
            "content": "Netflix added 8.8 million paid subscribers in Q3, bringing total to 247.2 million. Revenue was $8.5 billion, up 7.7% year-over-year. Operating margin improved to 22.4%.",
            "metadata": {"company": "Netflix", "quarter": "Q3 2023", "document_type": "earnings_report", "date": "2023-10-18"},
            "sentiment": "positive",
            "document_type": "earnings_report"
        },
        {
            "title": "Tesla Q2 2023 Financial Results",
            "content": "Tesla reported revenue of $24.9 billion, up 47% year-over-year. Vehicle deliveries were 466,140 units. Energy storage deployments increased 222% to 3.7 GWh.",
            "metadata": {"company": "Tesla", "quarter": "Q2 2023", "document_type": "earnings_report", "date": "2023-07-19"},
            "sentiment": "positive",
            "document_type": "earnings_report"
        },
        {
            "title": "Bank of America Annual Report 2023",
            "content": "Bank of America reported net income of $26.5 billion for 2023. Total revenue was $98.6 billion. The bank's efficiency ratio improved to 60.1%.",
            "metadata": {"company": "Bank of America", "year": "2023", "document_type": "annual_report", "date": "2023-03-15"},
            "sentiment": "neutral",
            "document_type": "annual_report"
        }
    ]
    
    # Combine all samples
    all_samples = sample_docs + additional_samples
    
    # Convert to DataFrame
    df = pd.DataFrame(all_samples)
    
    return df


def train_sentiment_model(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train and evaluate sentiment analysis model.
    
    Args:
        df: DataFrame with document data
        
    Returns:
        Dictionary with training results
    """
    logger.info("Training sentiment analysis model...")
    
    # Prepare data
    texts = df['content'].tolist()
    labels = df['sentiment'].tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Initialize model
    sentiment_analyzer = FinancialSentimentAnalyzer()
    
    # Make predictions on test set
    predictions = []
    for text in X_test:
        result = sentiment_analyzer.analyze_sentiment(text)
        predictions.append(result['sentiment'])
    
    # Evaluate
    evaluator = NLPEvaluator()
    metrics = evaluator.evaluate_sentiment_analysis(predictions, y_test)
    
    logger.info(f"Sentiment Analysis Results:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
    logger.info(f"  F1 Macro: {metrics['f1_macro']:.3f}")
    
    return {
        'model': 'sentiment_analysis',
        'metrics': metrics,
        'predictions': predictions,
        'ground_truth': y_test
    }


def train_document_classifier(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train and evaluate document classification model.
    
    Args:
        df: DataFrame with document data
        
    Returns:
        Dictionary with training results
    """
    logger.info("Training document classification model...")
    
    # Prepare data
    texts = df['content'].tolist()
    labels = df['document_type'].tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Initialize model
    classifier = FinancialDocumentClassifier()
    
    # Make predictions on test set
    predictions = []
    for text in X_test:
        result = classifier.classify_document(text)
        predictions.append(result['predicted_type'])
    
    # Evaluate
    evaluator = NLPEvaluator()
    metrics = evaluator.evaluate_document_classification(predictions, y_test)
    
    logger.info(f"Document Classification Results:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
    logger.info(f"  F1 Macro: {metrics['macro_avg_f1']:.3f}")
    
    return {
        'model': 'document_classification',
        'metrics': metrics,
        'predictions': predictions,
        'ground_truth': y_test
    }


def evaluate_ner_performance(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate NER performance on sample data.
    
    Args:
        df: DataFrame with document data
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info("Evaluating NER performance...")
    
    ner = FinancialNER()
    processor = FinancialDocumentProcessor()
    
    # Process documents and extract entities
    all_predictions = []
    all_ground_truth = []
    
    for _, row in df.iterrows():
        # Extract entities using our models
        financial_entities = ner.extract_financial_entities(row['content'])
        named_entities = processor.extract_named_entities(row['content'])
        
        # Combine entities
        combined_entities = []
        for entity_type, entities in financial_entities.items():
            for entity in entities:
                combined_entities.append((entity, entity_type))
        
        all_predictions.append(combined_entities)
        
        # Create ground truth (simplified for demo)
        ground_truth_entities = []
        if '$' in row['content']:
            # Extract money amounts as ground truth
            import re
            money_matches = re.findall(r'\$\d+(?:\.\d+)?\s?(?:billion|million|thousand|B|M|K)', row['content'])
            for match in money_matches:
                ground_truth_entities.append((match, 'MONEY'))
        
        all_ground_truth.append(ground_truth_entities)
    
    # Evaluate
    evaluator = NLPEvaluator()
    metrics = evaluator.evaluate_ner(all_predictions, all_ground_truth)
    
    logger.info(f"NER Evaluation Results:")
    logger.info(f"  Exact Match Accuracy: {metrics['exact_match_accuracy']:.3f}")
    logger.info(f"  Total Entities: {metrics['total_entities']}")
    
    return {
        'model': 'ner',
        'metrics': metrics,
        'predictions': all_predictions,
        'ground_truth': all_ground_truth
    }


def evaluate_summarization_performance(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate summarization performance.
    
    Args:
        df: DataFrame with document data
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info("Evaluating summarization performance...")
    
    summarizer = FinancialTextSummarizer()
    
    summaries = []
    original_lengths = []
    summary_lengths = []
    
    for _, row in df.iterrows():
        text = row['content']
        summary = summarizer.extractive_summarize(text, num_sentences=2)
        
        summaries.append(summary)
        original_lengths.append(len(text.split()))
        summary_lengths.append(len(summary.split()))
    
    # Calculate compression ratio
    compression_ratios = [orig_len / sum_len if sum_len > 0 else 0 
                         for orig_len, sum_len in zip(original_lengths, summary_lengths)]
    
    avg_compression = np.mean(compression_ratios)
    
    logger.info(f"Summarization Results:")
    logger.info(f"  Average Compression Ratio: {avg_compression:.2f}")
    logger.info(f"  Average Original Length: {np.mean(original_lengths):.1f} words")
    logger.info(f"  Average Summary Length: {np.mean(summary_lengths):.1f} words")
    
    return {
        'model': 'summarization',
        'avg_compression_ratio': avg_compression,
        'avg_original_length': np.mean(original_lengths),
        'avg_summary_length': np.mean(summary_lengths),
        'summaries': summaries
    }


def create_model_comparison(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a model comparison leaderboard.
    
    Args:
        results: List of model results
        
    Returns:
        DataFrame with model comparison
    """
    comparison_evaluator = ModelComparisonEvaluator()
    
    # Prepare results for comparison
    model_results = {}
    for result in results:
        model_name = result['model']
        if 'metrics' in result:
            model_results[model_name] = result['metrics']
    
    # Create leaderboard
    leaderboard = comparison_evaluator.create_leaderboard(model_results, 'accuracy')
    
    return leaderboard


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Financial Document Analysis models')
    parser.add_argument('--output-dir', type=str, default='assets', 
                       help='Output directory for results')
    parser.add_argument('--models', nargs='+', 
                       default=['sentiment', 'classification', 'ner', 'summarization'],
                       help='Models to train')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info("Starting Financial Document Analysis training...")
    
    # Create sample dataset
    logger.info("Creating sample dataset...")
    df = create_sample_dataset()
    logger.info(f"Created dataset with {len(df)} documents")
    
    # Train models
    results = []
    
    if 'sentiment' in args.models:
        sentiment_results = train_sentiment_model(df)
        results.append(sentiment_results)
    
    if 'classification' in args.models:
        classification_results = train_document_classifier(df)
        results.append(classification_results)
    
    if 'ner' in args.models:
        ner_results = evaluate_ner_performance(df)
        results.append(ner_results)
    
    if 'summarization' in args.models:
        summarization_results = evaluate_summarization_performance(df)
        results.append(summarization_results)
    
    # Create model comparison
    logger.info("Creating model comparison...")
    leaderboard = create_model_comparison(results)
    
    # Save results
    logger.info(f"Saving results to {output_dir}")
    
    # Save leaderboard
    leaderboard.to_csv(output_dir / 'model_leaderboard.csv', index=False)
    
    # Save detailed results
    for result in results:
        model_name = result['model']
        if 'metrics' in result:
            metrics_df = pd.DataFrame([result['metrics']])
            metrics_df.to_csv(output_dir / f'{model_name}_metrics.csv', index=False)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("TRAINING SUMMARY")
    logger.info("="*50)
    
    print(leaderboard[['model', 'rank', 'primary_score']].to_string(index=False))
    
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
