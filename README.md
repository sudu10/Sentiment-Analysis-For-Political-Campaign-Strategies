# Political Sentiment Analysis & Campaign Strategy Generator

## Overview
This comprehensive toolkit provides data-driven insights for political campaigns through advanced sentiment analysis of social media content. Built on BERT-based natural language processing, the system consists of two powerful components that analyze political discourse to generate actionable campaign strategies tailored to specific regions, topics, and political contexts.

## Key Components

### DSproject_BERT.py: Sentiment Analysis Foundation
The core sentiment analysis engine that provides:
- BERT-based sentiment classification of political tweets
- Interactive Streamlit dashboard with comprehensive visualization capabilities
- Temporal sentiment trend analysis of political discourse
- Confusion matrix and model performance metrics
- Network analysis of political entities and relationships
- Export capabilities for detailed analysis results

### XAI.py: Campaign Intelligence Platform
The Explainable AI dashboard provides a powerful interface for campaign strategists to:
- Perform location-specific sentiment analysis on political discourse
- Generate data-driven campaign recommendations based on regional sentiment patterns
- Filter and analyze sentiment by political party mentions
- Conduct keyword and topic-based analysis for targeted messaging
- Export comprehensive strategy documents in PDF format

## Features

### Deep Learning Sentiment Analysis
- Leverages BERT (Bidirectional Encoder Representations from Transformers) for nuanced sentiment classification
- Three-class sentiment categorization: Positive (2), Neutral (1), and Negative (0)
- Batch processing capabilities for efficient analysis of large datasets

### Interactive Visualization Dashboard
- Regional sentiment heatmaps highlighting geographic variations
- Temporal trend analysis tracking sentiment evolution over time
- Party-specific sentiment breakdowns for competitive analysis
- Topic-based filtering to focus on specific campaign issues
- Network visualization of political relationships (DSproject_BERT.py)

### Automated Strategy Generation
- Data-driven campaign recommendations based on sentiment distributions
- Region-specific tactical suggestions
- Party-targeted competitive analysis
- Topic-focused messaging recommendations
- Exportable PDF reports for stakeholder distribution

### Advanced Filtering Capabilities
- Geographic filtering for targeted regional analysis
- Sentiment-based filtering to identify positive/negative clusters
- Keyword filtering to track specific issues or topics
- Party-based filtering for opposition research

## Technical Implementation

### Model Architecture
- BERT-base-uncased transformer model
- Fine-tuned sequence classification for political sentiment detection
- Custom TweetDataset class for efficient data processing
- CUDA acceleration support for performance optimization

### Data Visualization
- Interactive Plotly visualizations for exploratory analysis
- Dynamic heatmaps showing sentiment distribution across regions
- Time-series tracking of sentiment evolution
- Comparative bar charts for multi-dimensional analysis
- Confusion matrices for model performance evaluation (DSproject_BERT.py)

### Report Generation
- Automated PDF generation via FPDF
- Structured campaign strategy recommendations
- Data-driven tactical suggestions
- Exportable formats for executive presentations

## Data Requirements

### For DSproject_BERT.py
- **Format**: CSV
- **Required Column**: 'tweet' (text content)
- **Sample Structure**:
  ```
  tweet
  "Great policy announcement by the government!"
  "Disappointed with the new legislation"
  ```

### For XAI.py
- **Format**: CSV
- **Required Columns**: 'tweet', 'location'
- **Sample Structure**:
  ```
  tweet,location,date
  "Campaign rally was amazing!",Delhi
  "Poor infrastructure in our area",Mumbai
  ```

## Usage Instructions

### Setup
1. Ensure Python 3.7+ is installed
2. Install required dependencies:
   ```
   pip install streamlit pandas torch transformers matplotlib fpdf plotly scikit-learn
   ```
3. Download or clone the repository

### Running DSproject_BERT.py
For comprehensive sentiment analysis:
```bash
streamlit run DSproject_BERT.py
```

### Running XAI.py
For campaign strategy generation:
```bash
streamlit run XAI.py
```

### Data Analysis Workflow
1. Upload CSV data containing tweets and required information
2. View automated sentiment analysis results across multiple dimensions
3. Filter by location, sentiment, or keywords for targeted insight
4. Generate region-specific campaign strategies
5. Export results as CSV or PDF reports

## Sentiment Interpretation Guide

### Sentiment Classification
- **Positive (2)**: Content expressing support, approval, or enthusiasm
- **Neutral (1)**: Content with balanced or ambiguous sentiment
- **Negative (0)**: Content expressing criticism, disapproval, or dissatisfaction

### Strategy Recommendations
The system generates strategies through rule-based analysis of sentiment patterns:
- High positive sentiment (>40%): Amplification strategies
- High negative sentiment (>30%): Mitigation and counter-messaging strategies
- High neutral sentiment (>20%): Engagement and persuasion strategies

## Advanced Features

### Party-Based Analysis
The system detects mentions of major political parties including:
- BJP, INC, AAP, TMC, CPI, CPI(M), NCP, SP, BSP, DMK, AIADMK, JD(U), RJD, 
  Shiv Sena, Akali Dal, YSRCP, BJD, RLD, LJP, AIMIM

### Topic Modeling
- Automatic extraction of trending topics from corpus
- Custom keyword filtering for issue-specific analysis
- Identification of extreme sentiment expressions by topic

### Network Analysis (DSproject_BERT.py)
- Visualization of relationships between political entities
- Identification of key influencers and communication patterns
- Cluster analysis of political discourse networks

## Limitations and Considerations
- Model performance depends on the quality and representativeness of input data
- Regional language variations may impact sentiment detection accuracy
- Social media data may not represent broader public opinion
- Strategies should be validated with additional research methods

## Future Development
- Integration with real-time social media APIs
- Multilingual support for regional language analysis
- Enhanced geospatial visualization capabilities
- Comparative analysis across multiple campaigns

## References
Khan, I. U., & Khan, M. U. S. (2024). Social Media Profiling for Political Affiliation Detection. *Human-Centric Intelligent Systems*, 4, 437-446. https://doi.org/10.1007/s44230-024-00078-y

## License
This project is licensed under the MIT License.
