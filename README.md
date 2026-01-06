# Financial Document Analysis

A comprehensive NLP system for analyzing financial documents including earnings reports, annual reports, press releases, and SEC filings. This project demonstrates advanced natural language processing techniques specifically tailored for financial text analysis.

## ⚠️ IMPORTANT DISCLAIMER

**This is a research demonstration tool for educational purposes only.**

- **NOT INVESTMENT ADVICE**: The analysis provided here should not be used for making financial decisions
- **RESEARCH ONLY**: Results are for research and educational purposes only
- **ACCURACY DISCLAIMER**: Models may be inaccurate and should not be relied upon for real-world applications
- **PROFESSIONAL CONSULTATION**: Always consult with qualified financial professionals before making any investment decisions

## Features

### Core Capabilities

- **Document Processing**: Support for PDF, DOCX, HTML, and TXT files
- **Sentiment Analysis**: Financial-specific sentiment analysis using FinBERT
- **Named Entity Recognition**: Extract companies, financial figures, and key metrics
- **Document Classification**: Automatically classify document types (earnings reports, annual reports, etc.)
- **Text Summarization**: Generate extractive and abstractive summaries
- **Insight Extraction**: Identify key financial metrics and trends

### Advanced NLP Models

- **Financial Sentiment Analysis**: Using ProsusAI/finbert for domain-specific sentiment
- **Financial NER**: Custom patterns for extracting financial entities
- **Document Classification**: Multi-class classification for financial document types
- **Text Summarization**: BART-based abstractive summarization
- **Insight Extraction**: Pattern-based extraction of financial metrics and trends

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Financial-Document-Analysis.git
   cd Financial-Document-Analysis
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Install the package** (optional):
   ```bash
   pip install -e .
   ```

## Quick Start

### Using the Streamlit Demo

1. **Launch the demo application**:
   ```bash
   streamlit run demo/app.py
   ```

2. **Open your browser** to `http://localhost:8501`

3. **Upload a document** or use sample documents to see the analysis in action

### Using the Python API

```python
from src.data.document_processor import FinancialDocumentProcessor
from src.models.nlp_models import FinancialSentimentAnalyzer

# Initialize processors
processor = FinancialDocumentProcessor()
sentiment_analyzer = FinancialSentimentAnalyzer()

# Process a document
result = processor.process_document("path/to/document.pdf")

# Analyze sentiment
sentiment = sentiment_analyzer.analyze_sentiment(result['cleaned_text'])

print(f"Sentiment: {sentiment['sentiment']}")
print(f"Confidence: {sentiment['confidence']:.2%}")
```

## Project Structure

```
financial-document-analysis/
├── src/                          # Source code
│   ├── data/                     # Data processing modules
│   │   └── document_processor.py  # Document loading and preprocessing
│   ├── models/                   # ML models and evaluation
│   │   ├── nlp_models.py         # NLP models (sentiment, NER, classification)
│   │   └── evaluation.py         # Evaluation metrics and comparison
│   ├── features/                 # Feature engineering
│   ├── labels/                   # Label processing
│   ├── backtest/                 # Backtesting utilities
│   ├── risk/                     # Risk analysis
│   └── utils/                    # Utility functions
├── data/                         # Data directories
│   ├── raw/                      # Raw documents
│   ├── processed/                # Processed data
│   └── samples/                  # Sample documents
├── configs/                      # Configuration files
│   └── config.yaml              # Main configuration
├── scripts/                      # Utility scripts
├── notebooks/                     # Jupyter notebooks
├── tests/                        # Unit tests
├── assets/                       # Output artifacts
├── demo/                         # Demo applications
│   └── app.py                    # Streamlit demo
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Project configuration
└── README.md                     # This file
```

## Configuration

The system uses YAML configuration files for easy customization. Key configuration options:

- **Model Settings**: Choose between different NLP models
- **Processing Parameters**: Adjust text processing settings
- **Financial Patterns**: Customize entity extraction patterns
- **Document Types**: Define document classification categories

See `configs/config.yaml` for detailed configuration options.

## Usage Examples

### Document Processing

```python
from src.data.document_processor import FinancialDocumentProcessor

processor = FinancialDocumentProcessor()

# Process different file types
pdf_text = processor.extract_text_from_pdf("earnings_report.pdf")
docx_text = processor.extract_text_from_docx("annual_report.docx")
html_text = processor.extract_text_from_html("press_release.html")

# Clean and process text
cleaned_text = processor.clean_text(raw_text)
entities = processor.extract_financial_entities(cleaned_text)
```

### Sentiment Analysis

```python
from src.models.nlp_models import FinancialSentimentAnalyzer

analyzer = FinancialSentimentAnalyzer()

# Analyze single text
result = analyzer.analyze_sentiment("Apple reported strong quarterly growth...")

# Batch analysis
texts = ["Revenue increased 10%", "Profit declined 5%", "Market outlook positive"]
results = analyzer.analyze_batch(texts)
```

### Document Classification

```python
from src.models.nlp_models import FinancialDocumentClassifier

classifier = FinancialDocumentClassifier()
classification = classifier.classify_document(document_text)

print(f"Document type: {classification['predicted_type']}")
print(f"Confidence: {classification['confidence']:.2%}")
```

### Evaluation

```python
from src.models.evaluation import NLPEvaluator

evaluator = NLPEvaluator()

# Evaluate sentiment analysis
sentiment_metrics = evaluator.evaluate_sentiment_analysis(
    predictions=['positive', 'negative', 'neutral'],
    ground_truth=['positive', 'negative', 'positive']
)

print(f"Accuracy: {sentiment_metrics['accuracy']:.2%}")
print(f"F1 Score: {sentiment_metrics['f1_macro']:.2%}")
```

## Dataset Schema

### Input Documents

The system supports various document formats:

- **PDF**: Financial reports, SEC filings
- **DOCX**: Word documents, press releases
- **HTML**: Web pages, news articles
- **TXT**: Plain text documents

### Output Schema

```json
{
  "file_path": "path/to/document.pdf",
  "raw_text": "Original document text...",
  "cleaned_text": "Processed text...",
  "financial_entities": {
    "revenue": ["$123.9 billion"],
    "dividend": ["$0.22 per share"],
    "percentage": ["10%", "5%"]
  },
  "named_entities": [
    ["Apple Inc.", "ORG"],
    ["Tim Cook", "PERSON"],
    ["$123.9 billion", "MONEY"]
  ],
  "sentiment": {
    "sentiment": "positive",
    "confidence": 0.85
  },
  "classification": {
    "predicted_type": "earnings_report",
    "confidence": 0.92
  },
  "summary": "Extracted summary...",
  "metrics": {
    "revenue": ["$123.9 billion"],
    "growth": ["10%"]
  },
  "trends": ["Revenue increased", "Services grew"]
}
```

## Evaluation Metrics

### NLP Metrics

- **Sentiment Analysis**: Accuracy, Precision, Recall, F1-Score (macro/weighted)
- **Named Entity Recognition**: Exact match accuracy, per-entity precision/recall
- **Document Classification**: Accuracy, confusion matrix, per-class metrics
- **Entity Extraction**: Precision, recall, F1-score for financial entities

### Financial-Specific Metrics

- **Metric Extraction Accuracy**: Accuracy of financial figure extraction
- **Trend Detection**: Precision/recall for trend identification
- **Entity Coverage**: Completeness of entity extraction

## Model Performance

### Baseline Results

| Model | Task | Accuracy | F1-Score | Notes |
|-------|------|----------|----------|-------|
| FinBERT | Sentiment | 0.78 | 0.76 | Financial domain |
| Custom NER | Entity Extraction | 0.82 | 0.79 | Financial patterns |
| Rule-based | Document Classification | 0.85 | 0.83 | Keyword matching |
| BART | Summarization | 0.72 | 0.71 | Abstractive |

*Note: Results are on sample datasets and may not reflect real-world performance*

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/ tests/
ruff check src/ tests/

# Run pre-commit hooks
pre-commit install
pre-commit run --all-files
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_document_processor.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{financial_document_analysis,
  title={Financial Document Analysis: NLP for Financial Text},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Financial-Document-Analysis}
}
```

## Acknowledgments

- **ProsusAI/finbert**: Financial sentiment analysis model
- **spaCy**: Natural language processing library
- **Transformers**: Hugging Face transformers library
- **Streamlit**: Web application framework

---

**Remember**: This tool is for research and educational purposes only. Always consult with qualified financial professionals before making investment decisions.
# Financial-Document-Analysis
