# Political Sentiment Analysis & Campaign Strategy Generator

## Project Overview

This project consists of two main components:

1. **DSproject_BERT.py**: Advanced sentiment analysis dashboard for political tweets with comprehensive visualization capabilities
2. **XAI.py**: Explainable AI tool that generates actionable campaign strategies based on sentiment analysis

Inspired by research from Khan & Khan (2024) on political affiliation detection using BERT, this implementation extends their work with practical applications for political campaign strategy.

## Features

### DSproject_BERT.py
- BERT-based sentiment classification (Positive/Negative/Neutral)
- Interactive Streamlit dashboard with multiple visualization options
- Temporal sentiment trend analysis
- Confusion matrix and model performance metrics
- Network analysis of political entities
- Export capabilities for analysis results

### XAI.py
- Location-based sentiment analysis
- Automated campaign strategy generation
- Party-specific sentiment breakdown
- Topic modeling and keyword extraction
- PDF report generation for strategies
- Data export functionality

## Usage

### DSproject_BERT.py
For comprehensive sentiment analysis:
```bash
streamlit run DSproject_BERT.py
```

**Dataset Requirements:**
CSV file containing tweets with "text" as column name

### XAI.py
For campaign strategy generation:
```bash
streamlit run XAI.py
```

**Dataset Requirements:**
CSV file with columns: 'tweet' (required), 'location' (required)

## Dataset Information

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

## Technical Details

### Models
- BERT-base-uncased for sentiment classification
- Custom fine-tuning for political context
- SHAP values for explainability (XAI.py)

### Key Components
- **Sentiment Analysis Pipeline**: Tokenization → BERT Inference → Classification
- **Strategy Generation**: Rule-based recommendations from sentiment distributions
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Reporting**: PDF generation with FPDF

## Results Interpretation

### Sentiment Classification
- **Positive (2)**: Supportive of subject/party
- **Neutral (1)**: No clear stance
- **Negative (0)**: Critical of subject/party

### Strategy Recommendations
Strategies are generated based on:
- Regional sentiment distributions
- Party-specific sentiment patterns
- Topic prevalence in conversations

## References

Khan, I. U., & Khan, M. U. S. (2024). Social Media Profiling for Political Affiliation Detection. *Human-Centric Intelligent Systems*, 4, 437-446. https://doi.org/10.1007/s44230-024-00078-y

## License

This project is licensed under the MIT License.
