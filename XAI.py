import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import os
from sklearn.feature_extraction.text import CountVectorizer
import datetime
import plotly.express as px
import random

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class TweetDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors="pt"
        )
        return {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0)
        }

def inject_dummy_sentiments(texts):
    sentiments = []
    for _ in texts:
        sentiments.append(random.choices([0, 1, 2], weights=[0.4, 0.2, 0.4])[0])  # Balanced
    return sentiments        

@st.cache_data
def predict_sentiments(texts):
    model.eval()
    dataset = TweetDataset(texts)
    dataloader = DataLoader(dataset, batch_size=8)

    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            predictions.extend(preds)
    return predictions

def generate_campaign_strategies(data):
    sentiment_counts = data['sentiment'].value_counts(normalize=True)
    strategies = []

    if sentiment_counts.get(2, 0) > 0.4:
        strategies.append("Promote positive regional developments via local leaders.")
    if sentiment_counts.get(0, 0) > 0.3:
        strategies.append("Counter regional dissatisfaction with tangible actions.")
        strategies.append("Strengthen ground teams to correct misinformation.")
    if sentiment_counts.get(1, 0) > 0.2:
        strategies.append("Engage neutral audiences through constructive debates.")

    strategies.append("Adjust campaign tones for regional alignment.")
    return strategies

def generate_pdf(strategies):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, "Campaign Strategy Recommendations", align='C')
    pdf.ln()
    for i, strat in enumerate(strategies, 1):
        pdf.multi_cell(0, 10, f"{i}. {strat}")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    return temp_file.name

def extract_common_keywords(texts, top_n=20):
    vectorizer = CountVectorizer(stop_words='english', max_features=top_n)
    X = vectorizer.fit_transform(texts)
    keywords = vectorizer.get_feature_names_out()
    return sorted(keywords)

st.title("BRET-Based Campaign Sentiment Dashboard")
st.write("Upload a CSV with 'tweet', 'location', 'date', and optionally 'party' columns to get started.")

uploaded_file = st.file_uploader("Upload Tweet CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'tweet' in df.columns and 'location' in df.columns:
        with st.spinner("Analyzing sentiments..."):
            df['sentiment'] = predict_sentiments(df['tweet'].tolist())

        st.success("Sentiment analysis complete!")
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        df['sentiment_label'] = df['sentiment'].map(sentiment_map)

        st.sidebar.header("Filter Tweets")
        locations = st.sidebar.multiselect("Select Locations", df['location'].unique())
        sentiments = st.sidebar.multiselect("Select Sentiments", df['sentiment_label'].unique())
        keywords = extract_common_keywords(df['tweet'])
        keyword_filter = st.sidebar.multiselect("Select Keywords", keywords)

        filtered_df = df.copy()
        if locations:
            filtered_df = filtered_df[filtered_df['location'].isin(locations)]
        if sentiments:
            filtered_df = filtered_df[filtered_df['sentiment_label'].isin(sentiments)]
        if keyword_filter:
            keyword_pattern = '|'.join(keyword_filter)
            filtered_df = filtered_df[filtered_df['tweet'].str.contains(keyword_pattern, case=False)]

        if 'date' in df.columns:
            st.subheader("📆 Sentiment Trend Over Time")
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            time_df = df.dropna(subset=['date'])
            trend_data = time_df.groupby([time_df['date'].dt.date, 'sentiment_label']).size().unstack(fill_value=0)
            st.line_chart(trend_data)

        st.subheader("📍 Heatmap of Sentiment by Location")
        fig = px.density_heatmap(
            df,
            x='sentiment_label',
            y='location',
            histfunc='count',
            nbinsx=3,
            color_continuous_scale='Viridis',
            title="Sentiment Heatmap by Location"
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🧠 Generate Campaign Strategy by Location")
        selected_loc = st.selectbox("Select location for strategy recommendation", df['location'].unique())
        if st.button("Generate Strategy PDF"):
            loc_df = df[df['location'] == selected_loc]
            strat_list = generate_campaign_strategies(loc_df)
            pdf_path = generate_pdf(strat_list)
            with open(pdf_path, "rb") as f:
                st.download_button("Download Strategy PDF", f.read(), file_name=f"strategy_{selected_loc}.pdf")

        st.subheader("💾 Export Full Data with Sentiment")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "sentiment_annotated_tweets.csv", "text/csv")

        if 'date' in df.columns:
            st.subheader("📆 Sentiment Trend Over Time")
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            time_df = df.dropna(subset=['date'])
            trend_data = time_df.groupby([time_df['date'].dt.date, 'sentiment_label']).size().unstack(fill_value=0).reindex(columns=['Positive', 'Neutral', 'Negative'], fill_value=0)
            st.line_chart(trend_data)

        regions = df['location'].unique().tolist()
        selected_region = st.selectbox("Select Region to Analyze", regions)
        regional_df = df[df['location'] == selected_region]

        st.subheader(f"📊 Sentiment Breakdown - {selected_region}")
        st.bar_chart(regional_df['sentiment_label'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0))

        st.subheader(f"🎯 Campaign Strategy - {selected_region}")
        strategies = generate_campaign_strategies(regional_df)
        for i, strat in enumerate(strategies, 1):
            st.markdown(f"**{i}.** {strat}")

        if st.button("Download Region Strategy as PDF"):
            pdf_path = generate_pdf(strategies)
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", f, f"campaign_strategies_{selected_region}.pdf", "application/pdf")
            os.remove(pdf_path)

        st.subheader("🇮🇳 Party-Based Filtering and Strategy")
        top_parties = [
            "BJP", "INC", "AAP", "TMC", "CPI", "CPI(M)", "NCP", "SP", "BSP", "DMK",
            "AIADMK", "JD(U)", "RJD", "Shiv Sena", "Akali Dal", "YSRCP", "BJD", "RLD", "LJP", "AIMIM"
        ]
        present_parties = df['tweet'].dropna().apply(lambda x: [p for p in top_parties if p.lower() in x.lower()])
        df['party'] = present_parties.apply(lambda x: x[0] if x else None)

        available_parties = sorted(df['party'].dropna().unique().tolist())
        selected_party = st.selectbox("Select a Political Party", available_parties)

        if selected_party:
            party_df = df[df['party'] == selected_party]
            st.write(f"Found {len(party_df)} tweets mentioning {selected_party}")
            if not party_df.empty:
                st.subheader(f"📍 Sentiment by Location ({selected_party})")
                party_loc = party_df.groupby(['location', 'sentiment_label']).size().unstack(fill_value=0).reindex(columns=['Positive', 'Neutral', 'Negative'], fill_value=0)
                st.dataframe(party_loc)
                st.bar_chart(party_loc)

                st.subheader(f"🎯 Strategy for {selected_party}")
                party_strategies = generate_campaign_strategies(party_df)
                for i, strat in enumerate(party_strategies, 1):
                    st.markdown(f"**{i}.** {strat}")

                if st.button("Download Party Strategy as PDF"):
                    pdf_path = generate_pdf(party_strategies)
                    with open(pdf_path, "rb") as f:
                        st.download_button("Download Party PDF", f, f"campaign_strategies_{selected_party}.pdf", "application/pdf")
                    os.remove(pdf_path)

                st.subheader("📝 Most Extreme Tweet about Party")
                extreme_idx = party_df['sentiment'].idxmin() if 0 in party_df['sentiment'].values else party_df['sentiment'].idxmax()
                st.markdown(f"> *{party_df.loc[extreme_idx, 'tweet']}*")

        st.subheader("📌 Topic-Based Filtering and Strategy")
        common_keywords = extract_common_keywords(df['tweet'].dropna().tolist())
        keyword = st.selectbox("Select a common topic or enter your own:", [""] + common_keywords)
        custom_keyword = st.text_input("Or type your own topic/keyword:")
        final_keyword = custom_keyword if custom_keyword else keyword

        if final_keyword:
            filtered_df = df[df['tweet'].str.contains(final_keyword, case=False, na=False)]
            st.write(f"Found {len(filtered_df)} tweets related to '{final_keyword}'")

            if not filtered_df.empty:
                st.subheader("📍 Sentiment by Location (Topic)")
                topic_sent = filtered_df.groupby(['location', 'sentiment_label']).size().unstack(fill_value=0).reindex(columns=['Positive', 'Neutral', 'Negative'], fill_value=0)
                st.dataframe(topic_sent)
                st.bar_chart(topic_sent)

                st.subheader("🎯 Topic-Based Strategy")
                topic_strategies = generate_campaign_strategies(filtered_df)
                for i, strat in enumerate(topic_strategies, 1):
                    st.markdown(f"**{i}.** {strat}")

                if st.button("Download Topic Strategy as PDF"):
                    pdf_path = generate_pdf(topic_strategies)
                    with open(pdf_path, "rb") as f:
                        st.download_button("Download Topic PDF", f, f"campaign_strategies_{final_keyword}.pdf", "application/pdf")
                    os.remove(pdf_path)

                st.subheader("📝 Most Extreme Tweet on Topic")
                extreme_idx = filtered_df['sentiment'].idxmin() if 0 in filtered_df['sentiment'].values else filtered_df['sentiment'].idxmax()
                st.markdown(f"> *{filtered_df.loc[extreme_idx, 'tweet']}*")
    else:
        st.error("CSV must contain both 'tweet' and 'location' columns.")
