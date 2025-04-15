# Sentiment-Analysis-For-Political-Campaign-Strategies

```
# BRET-Based Campaign Sentiment Dashboard

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-yellow?style=for-the-badge&logo=huggingface&logoColor=white)

## 📌 Overview
A political campaign sentiment analysis tool that:
- Classifies tweet sentiments using BERT (Positive/Neutral/Negative)
- Visualizes geographical and temporal trends
- Generates PDF campaign strategies
- Analyzes party-specific and topic-based sentiment

2. Run the app:
```bash
streamlit run app.py
```

3. Upload a CSV with these columns:
   - `tweet` (text content)
   - `location` (geographical data)

##  Features

### Sentiment Analysis
- BERT-based classification (Positive/Neutral/Negative)
- Batch processing for efficiency
- GPU acceleration support

### Geographical Insights
- Interactive heatmap by location
- Region-specific strategy generation
- Location filtering capabilities

### Temporal Analysis
- Daily sentiment trends
- Time-based filtering
- Historical pattern identification

### Strategy Generation
- Automated PDF report creation
- Context-aware recommendations
- Downloadable strategies

### Party Analysis
- Detects mentions of 20+ Indian political parties
- Party-specific sentiment breakdown
- Custom strategy generation

### Topic Modeling
- Automatic keyword extraction
- Topic-based sentiment analysis
- Custom keyword filtering

## Example CSV Format

| tweet                          | location    |
|--------------------------------|-------------|
| "Great policy announcement!"   | Delhi       |
| "Disappointed with leadership" | Mumbai      |

## 🛠️ Technical Details

### Model Architecture
- BERT-base-uncased
- 3-class classification (Negative/Neutral/Positive)
- Max sequence length: 128 tokens
