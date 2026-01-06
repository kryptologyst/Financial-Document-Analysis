"""
Financial Document Analysis - Streamlit Demo

Interactive web application for demonstrating financial document analysis
capabilities including sentiment analysis, entity extraction, document
classification, and summarization.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.document_processor import FinancialDocumentProcessor, create_sample_financial_documents
from src.models.nlp_models import (
    FinancialSentimentAnalyzer,
    FinancialNER,
    FinancialDocumentClassifier,
    FinancialTextSummarizer,
    FinancialInsightExtractor
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Financial Document Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
DISCLAIMER = """
**⚠️ IMPORTANT DISCLAIMER**

This is a research demonstration tool for educational purposes only. 
The analysis provided here is NOT investment advice and should not be used 
for making financial decisions. The models may be inaccurate and results 
are for research/educational purposes only. Always consult with qualified 
financial professionals before making any investment decisions.
"""

def initialize_session_state():
    """Initialize session state variables."""
    if 'processed_documents' not in st.session_state:
        st.session_state.processed_documents = []
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}


def load_models():
    """Load NLP models (with caching)."""
    if 'models_loaded' not in st.session_state:
        with st.spinner("Loading NLP models..."):
            try:
                st.session_state.document_processor = FinancialDocumentProcessor()
                st.session_state.sentiment_analyzer = FinancialSentimentAnalyzer()
                st.session_state.ner = FinancialNER()
                st.session_state.classifier = FinancialDocumentClassifier()
                st.session_state.summarizer = FinancialTextSummarizer()
                st.session_state.insight_extractor = FinancialInsightExtractor()
                st.session_state.models_loaded = True
                st.success("Models loaded successfully!")
            except Exception as e:
                st.error(f"Error loading models: {e}")
                st.session_state.models_loaded = False


def process_uploaded_file(uploaded_file) -> Dict[str, Any]:
    """Process an uploaded file."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Process the document
        processor = st.session_state.document_processor
        result = processor.process_document(tmp_path)
        
        # Clean up temporary file
        Path(tmp_path).unlink()
        
        return result
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return {}


def analyze_document(text: str) -> Dict[str, Any]:
    """Perform comprehensive analysis on document text."""
    results = {}
    
    try:
        # Sentiment analysis
        sentiment_result = st.session_state.sentiment_analyzer.analyze_sentiment(text)
        results['sentiment'] = sentiment_result
        
        # NER
        financial_entities = st.session_state.ner.extract_financial_entities(text)
        companies = st.session_state.ner.extract_companies(text)
        results['financial_entities'] = financial_entities
        results['companies'] = companies
        
        # Document classification
        classification = st.session_state.classifier.classify_document(text)
        results['classification'] = classification
        
        # Summarization
        summary = st.session_state.summarizer.extractive_summarize(text)
        results['summary'] = summary
        
        # Insight extraction
        metrics = st.session_state.insight_extractor.extract_metrics(text)
        trends = st.session_state.insight_extractor.extract_trends(text)
        results['metrics'] = metrics
        results['trends'] = trends
        
    except Exception as e:
        st.error(f"Error in analysis: {e}")
        results['error'] = str(e)
    
    return results


def display_sentiment_analysis(results: Dict[str, Any]):
    """Display sentiment analysis results."""
    st.subheader("📊 Sentiment Analysis")
    
    sentiment_data = results.get('sentiment', {})
    if sentiment_data and 'error' not in sentiment_data:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sentiment = sentiment_data.get('sentiment', 'neutral')
            confidence = sentiment_data.get('confidence', 0.5)
            
            # Color coding for sentiment
            if sentiment == 'positive':
                color = '#28a745'
                emoji = '📈'
            elif sentiment == 'negative':
                color = '#dc3545'
                emoji = '📉'
            else:
                color = '#ffc107'
                emoji = '➡️'
            
            st.metric(
                label=f"{emoji} Overall Sentiment",
                value=sentiment.title(),
                delta=f"{confidence:.1%} confidence"
            )
        
        with col2:
            st.metric(
                label="Confidence Score",
                value=f"{confidence:.1%}",
                delta=None
            )
        
        with col3:
            # Sentiment distribution visualization
            sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
            sentiment_counts[sentiment] = 1
            
            fig = px.pie(
                values=list(sentiment_counts.values()),
                names=list(sentiment_counts.keys()),
                title="Sentiment Distribution",
                color_discrete_map={
                    'positive': '#28a745',
                    'neutral': '#ffc107', 
                    'negative': '#dc3545'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Sentiment analysis not available")


def display_entity_extraction(results: Dict[str, Any]):
    """Display entity extraction results."""
    st.subheader("🏷️ Entity Extraction")
    
    financial_entities = results.get('financial_entities', {})
    companies = results.get('companies', [])
    
    if financial_entities or companies:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Financial Entities:**")
            for entity_type, entities in financial_entities.items():
                if entities:
                    st.write(f"- **{entity_type.title()}**: {', '.join(entities)}")
        
        with col2:
            st.write("**Companies:**")
            if companies:
                for company in companies:
                    st.write(f"- {company}")
            else:
                st.write("No companies detected")
        
        # Entity visualization
        if financial_entities:
            entity_counts = {k: len(v) for k, v in financial_entities.items() if v}
            if entity_counts:
                fig = px.bar(
                    x=list(entity_counts.keys()),
                    y=list(entity_counts.values()),
                    title="Financial Entity Counts",
                    labels={'x': 'Entity Type', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No entities extracted")


def display_document_classification(results: Dict[str, Any]):
    """Display document classification results."""
    st.subheader("📋 Document Classification")
    
    classification = results.get('classification', {})
    if classification and 'error' not in classification:
        predicted_type = classification.get('predicted_type', 'unknown')
        confidence = classification.get('confidence', 0.0)
        all_scores = classification.get('all_scores', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Predicted Document Type",
                value=predicted_type.replace('_', ' ').title(),
                delta=f"{confidence:.1%} confidence"
            )
        
        with col2:
            # Classification scores visualization
            if all_scores:
                fig = px.bar(
                    x=list(all_scores.keys()),
                    y=list(all_scores.values()),
                    title="Classification Scores",
                    labels={'x': 'Document Type', 'y': 'Score'}
                )
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Document classification not available")


def display_summary(results: Dict[str, Any]):
    """Display document summary."""
    st.subheader("📝 Document Summary")
    
    summary = results.get('summary', '')
    if summary:
        st.write(summary)
    else:
        st.info("No summary available")


def display_insights(results: Dict[str, Any]):
    """Display extracted insights."""
    st.subheader("💡 Key Insights")
    
    metrics = results.get('metrics', {})
    trends = results.get('trends', [])
    
    if metrics:
        st.write("**Financial Metrics:**")
        for metric_name, metric_value in metrics.items():
            st.write(f"- **{metric_name.title()}**: {metric_value}")
    
    if trends:
        st.write("**Trends Detected:**")
        for trend in trends[:3]:  # Show top 3 trends
            st.write(f"- {trend}")
    
    if not metrics and not trends:
        st.info("No insights extracted")


def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">Financial Document Analysis</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown(f'<div class="warning-box">{DISCLAIMER}</div>', unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Load models
    load_models()
    
    if not st.session_state.get('models_loaded', False):
        st.error("Failed to load models. Please refresh the page.")
        return
    
    # Sidebar
    st.sidebar.title("📁 Document Input")
    
    # Input options
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Upload File", "Sample Documents", "Manual Text Input"]
    )
    
    document_text = ""
    document_title = ""
    
    if input_method == "Upload File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload a financial document",
            type=['txt', 'pdf', 'docx', 'html'],
            help="Supported formats: TXT, PDF, DOCX, HTML"
        )
        
        if uploaded_file:
            with st.spinner("Processing uploaded file..."):
                result = process_uploaded_file(uploaded_file)
                if result:
                    document_text = result.get('cleaned_text', '')
                    document_title = uploaded_file.name
                    st.session_state.processed_documents.append({
                        'title': document_title,
                        'text': document_text,
                        'source': 'upload'
                    })
    
    elif input_method == "Sample Documents":
        sample_docs = create_sample_financial_documents()
        
        selected_doc = st.sidebar.selectbox(
            "Select a sample document:",
            [doc['title'] for doc in sample_docs]
        )
        
        if selected_doc:
            doc_data = next(doc for doc in sample_docs if doc['title'] == selected_doc)
            document_text = doc_data['content']
            document_title = doc_data['title']
    
    elif input_method == "Manual Text Input":
        document_text = st.sidebar.text_area(
            "Enter financial text:",
            height=200,
            placeholder="Paste your financial document text here..."
        )
        document_title = "Manual Input"
    
    # Analysis section
    if document_text:
        st.markdown('<div class="success-box">Document loaded successfully!</div>', unsafe_allow_html=True)
        
        # Analysis controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            analyze_sentiment = st.checkbox("Sentiment Analysis", value=True)
        with col2:
            analyze_entities = st.checkbox("Entity Extraction", value=True)
        with col3:
            analyze_classification = st.checkbox("Document Classification", value=True)
        
        analyze_summary = st.checkbox("Document Summarization", value=True)
        analyze_insights = st.checkbox("Insight Extraction", value=True)
        
        # Run analysis
        if st.button("🔍 Analyze Document", type="primary"):
            with st.spinner("Analyzing document..."):
                analysis_results = analyze_document(document_text)
                st.session_state.analysis_results = analysis_results
            
            # Display results
            st.markdown("---")
            
            if analyze_sentiment:
                display_sentiment_analysis(analysis_results)
                st.markdown("---")
            
            if analyze_entities:
                display_entity_extraction(analysis_results)
                st.markdown("---")
            
            if analyze_classification:
                display_document_classification(analysis_results)
                st.markdown("---")
            
            if analyze_summary:
                display_summary(analysis_results)
                st.markdown("---")
            
            if analyze_insights:
                display_insights(analysis_results)
    
    else:
        st.info("👈 Please select an input method from the sidebar to begin analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Financial Document Analysis Demo | Research & Educational Use Only</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
