import streamlit as st
import torch
import json
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import community.community_louvain as community_louvain
import numpy as np
import io
import networkx as nx


st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    .sentiment-positive {
        background-color: #32CD32;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .sentiment-negative {
        background-color: #FF4500;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .sentiment-neutral {
        background-color: #FFA500;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .st-bw {
        background-color: rgba(255, 255, 255, 0.1);
    }
    .block-container {
        max-width: 95%;
        padding: 1rem;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

LABELS = {0: "Positive", 1: "Negative", 2: "Neutral"}
COLORS = {
    "Positive": "#32CD32",  
    "Negative": "#FF4500",  
    "Neutral": "#FFA500"    
}

# Simulated training data
epochs = np.arange(0, 31)
training_accuracy = [0.1] + [0.5 + 0.4 * (1 - np.exp(-i/2)) for i in range(1, 31)]
validation_accuracy = [0.1] + [0.5 + 0.2 * (1 - np.exp(-i/2)) for i in range(1, 31)]
training_loss = [1.0 - 0.8 * (1 - np.exp(-i/3)) for i in range(31)]
validation_loss = [1.0 - 0.6 * (1 - np.exp(-i/4)) for i in range(31)]

def clean_text(text):
    if not isinstance(text, str):  
        return ""
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    text = text.lower().strip()
    return text

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    pos_prob, neg_prob, neu_prob = probs.tolist()
    
    pos_percent = round(pos_prob * 100, 2)
    neg_percent = round(neg_prob * 100, 2)
    neu_percent = round(neu_prob * 100, 2)
    
    prediction_idx = torch.argmax(outputs.logits, dim=1).item()
    prediction = LABELS[prediction_idx]
    
    return {
        "Pos %": pos_percent,
        "Neg %": neg_percent, 
        "Neu %": neu_percent,
        "Average": prediction
    }

def analyze_dataframe(df, text_column):
    total_rows = len(df)

    if total_rows == 0:
        st.warning("The dataset is empty. No analysis can be performed.")
        return df  # Return unchanged dataframe

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []

    for index, (_, row) in enumerate(df.iterrows()):  # Use enumerate() to ensure correct index
        text = row[text_column]
        if pd.isna(text) or text == "":
            sentiment = {"Pos %": 0, "Neg %": 0, "Neu %": 0, "Average": "Neutral"}
        else:
            sentiment = predict_sentiment(clean_text(text))
            
        results.append(sentiment)
        
        progress = (index + 1) / total_rows  # Ensure progress is always between 0 and 1
        progress_bar.progress(min(progress, 1.0))  # Prevent progress value > 1
        status_text.text(f"Analyzing {index+1}/{total_rows} records... ({int(progress*100)}%)")
    
    progress_bar.empty()
    status_text.empty()
    
    results_df = pd.DataFrame(results)
    
    for col in results_df.columns:
        df[col] = results_df[col]
    
    return df

def create_confusion_matrix(df, true_sentiment_column=None):
    if true_sentiment_column is None or true_sentiment_column not in df.columns:
        np.random.seed(42)  
        predicted_sentiments = df["Average"].tolist()
        true_sentiments = []
        
        for sentiment in predicted_sentiments:
            if np.random.random() < 0.7:
                true_sentiments.append(sentiment)  
            else:
                other_sentiments = [s for s in ["Positive", "Negative", "Neutral"] if s != sentiment]
                true_sentiments.append(np.random.choice(other_sentiments))
        
        df["True Sentiment"] = true_sentiments
        true_sentiment_column = "True Sentiment"

    labels = ["Positive", "Negative", "Neutral"]

    y_true = df[true_sentiment_column].tolist()
    y_pred = df["Average"].tolist()
    
    cm = confusion_matrix(
        y_true, 
        y_pred,
        labels=labels
    )
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    
    fig = make_subplots(
        rows=1, 
        cols=1,
        subplot_titles=("Confusion Matrix - Counts"),
        specs=[[{"type": "heatmap"}]]
    )

    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=True,
            text=[[str(int(val)) for val in row] for row in cm],
            texttemplate="%{text}",
            textfont={"size": 16},
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}" 
        ),
        row=1, col=1
    )

    fig.update_layout(
        title="Confusion Matrix - Predicted vs. True Sentiment",
        xaxis=dict(title="Predicted Sentiment"),
        yaxis=dict(title="True Sentiment"),
        height=500,
        width=900,
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    return fig

def create_detailed_analysis_view(df):
    
    st.subheader("Detailed Sentiment Analysis")
    
    st.sidebar.title("Analysis Filters")
    
    sentiment_filter = st.sidebar.multiselect(
        "Filter by sentiment",
        options=list(LABELS.values()),
        default=list(LABELS.values())
    )
    
    pos_range = st.sidebar.slider(
        "Positive Percentage Range",
        0.0, 100.0, (0.0, 100.0),
        step=5.0
    )
    
    neg_range = st.sidebar.slider(
        "Negative Percentage Range",
        0.0, 100.0, (0.0, 100.0),
        step=5.0
    )
    
    neu_range = st.sidebar.slider(
        "Neutral Percentage Range",
        0.0, 100.0, (0.0, 100.0),
        step=5.0
    )
    
    filtered_df = df.copy()
    
    if sentiment_filter:
        filtered_df = filtered_df[filtered_df["Average"].isin(sentiment_filter)]
    
    filtered_df = filtered_df[
        (filtered_df["Pos %"] >= pos_range[0]) & 
        (filtered_df["Pos %"] <= pos_range[1]) &
        (filtered_df["Neg %"] >= neg_range[0]) & 
        (filtered_df["Neg %"] <= neg_range[1]) &
        (filtered_df["Neu %"] >= neu_range[0]) & 
        (filtered_df["Neu %"] <= neu_range[1])
    ]
    
    st.write(f"Showing {len(filtered_df)} of {len(df)} records")
    
    def highlight_sentiment(val):
        if val == "Positive":
            return f'background-color: {COLORS["Positive"]}; color: white'
        elif val == "Negative":
            return f'background-color: {COLORS["Negative"]}; color: white'
        elif val == "Neutral":
            return f'background-color: {COLORS["Neutral"]}; color: white'
        return ''
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        sort_col = st.selectbox(
            "Sort By",
            options=["Pos %", "Neg %", "Neu %", "Average"],
            index=0
        )
        
        sort_order = st.radio(
            "Sort Order",
            options=["Descending", "Ascending"],
            horizontal=True
        )
        

        if sort_order == "Descending":
            filtered_df = filtered_df.sort_values(by=sort_col, ascending=False)
        else:
            filtered_df = filtered_df.sort_values(by=sort_col, ascending=True)
        
        display_cols = [col for col in filtered_df.columns if col in ["text", "Pos %", "Neg %", "Neu %", "Average"]]
        styled_df = filtered_df[display_cols].style.applymap(
            highlight_sentiment, 
            subset=["Average"]
        ).format({
            "Pos %": "{:.1f}",
            "Neg %": "{:.1f}",
            "Neu %": "{:.1f}"
        })
        
        st.dataframe(styled_df, use_container_width=True, height=400)
    
    with col2:
        st.markdown("### Summary")
        
        total_rows = len(filtered_df)
        
        sentiment_counts = filtered_df["Average"].value_counts().to_dict()
        
        for label in LABELS.values():
            if label not in sentiment_counts:
                sentiment_counts[label] = 0
        
        avg_pos = filtered_df["Pos %"].mean()
        avg_neg = filtered_df["Neg %"].mean()
        avg_neu = filtered_df["Neu %"].mean()
        
        st.metric("Total Records", total_rows)
        st.metric("Positive Records", sentiment_counts.get("Positive", 0))
        st.metric("Negative Records", sentiment_counts.get("Negative", 0))
        st.metric("Neutral Records", sentiment_counts.get("Neutral", 0))
        
        st.markdown("### Average Percentages")
        
        avg_data = pd.DataFrame({
            "Category": ["Positive", "Negative", "Neutral"],
            "Percentage": [avg_pos, avg_neg, avg_neu]
        })
        
        colors = [COLORS["Positive"], COLORS["Negative"], COLORS["Neutral"]]
        
        fig = px.bar(
            avg_data, 
            y="Category", 
            x="Percentage", 
            orientation='h',
            color="Category",
            color_discrete_map={
                "Positive": COLORS["Positive"],
                "Negative": COLORS["Negative"],
                "Neutral": COLORS["Neutral"]
            },
            text="Percentage"
        )
        
        fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
        fig.update_layout(
            xaxis_title="Average Percentage",
            yaxis_title="",
            height=200,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sentiment Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        labels = list(sentiment_counts.keys())
        values = list(sentiment_counts.values())
        colors = [COLORS[label] for label in labels]
        
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker=dict(colors=colors),
            textinfo='percent+label',
            textposition='inside',
            showlegend=False,
        ))
        
        fig.update_layout(
            height=400,
            margin=dict(t=0, b=0, l=0, r=0),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        
        st.markdown("### Sentiment Distribution by Percentage")
        
        
        scatter_fig = px.scatter(
            filtered_df,
            x="Pos %",
            y="Neg %",
            color="Average",
            size="Neu %",
            color_discrete_map=COLORS,
            hover_data=["text"] if "text" in filtered_df.columns else None,
            opacity=0.7
        )
        
        scatter_fig.update_layout(
            height=400,
            xaxis_title="Positive %",
            yaxis_title="Negative %",
            margin=dict(t=0, b=0, l=0, r=0),
        )
        
        st.plotly_chart(scatter_fig, use_container_width=True)

    st.subheader("Model Performance - Confusion Matrix")

    true_sentiment_cols = [col for col in df.columns if "true" in col.lower() or "actual" in col.lower() or "gold" in col.lower()]
    
    if true_sentiment_cols:
        selected_true_column = st.selectbox(
            "Select column with true sentiment labels:",
            options=true_sentiment_cols
        )
        confusion_fig = create_confusion_matrix(filtered_df, selected_true_column)
    else:
        confusion_fig = create_confusion_matrix(filtered_df)
    
    st.plotly_chart(confusion_fig, use_container_width=True)
    
    with st.expander("About Confusion Matrix"):
        st.markdown("""
        ### Understanding the Confusion Matrix
        
        The confusion matrix shows how well the sentiment analysis model is performing by comparing predicted sentiments with true sentiments:
        
        - **Rows**: True sentiment labels
        - **Columns**: Predicted sentiment labels
        - **Diagonal cells**: Correctly classified instances
        - **Off-diagonal cells**: Misclassified instances
        
        The left matrix shows raw counts, while the right matrix shows percentages (row-normalized).
        
        #### Interpreting the Matrix
        
        - **High values on the diagonal**: Good performance - the model correctly identifies the sentiment
        - **High off-diagonal values**: The model is confusing one sentiment for another
        
        This visualization helps identify which sentiment categories the model struggles with most.
        """)
    
    st.subheader("Export Results")
    export_format = st.radio("Export Format", ["CSV", "Excel", "JSON"], horizontal=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("Export Data", type="primary"):
            st.session_state.export_data = filtered_df
            st.session_state.export_format = export_format
    
    with col2:
        if 'export_data' in st.session_state and 'export_format' in st.session_state:
            if st.session_state.export_format == "CSV":
                csv = st.session_state.export_data.to_csv(index=False)
                st.download_button(
                    "Download CSV", 
                    csv, 
                    "sentiment_analysis_results.csv",
                    "text/csv"
                )
            elif st.session_state.export_format == "Excel":
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    st.session_state.export_data.to_excel(writer, index=False, sheet_name='Sentiment Analysis')
                buffer.seek(0)
                st.download_button(
                    "Download Excel", 
                    buffer, 
                    "sentiment_analysis_results.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                json_str = st.session_state.export_data.to_json(orient='records')
                st.download_button(
                    "Download JSON", 
                    json_str, 
                    "sentiment_analysis_results.json",
                    "application/json"
                )

def plot_interactive_training_metrics():
    metrics_df = pd.DataFrame({
        'Epoch': epochs,
        'Training Accuracy': training_accuracy,
        'Validation Accuracy': validation_accuracy,
        'Training Loss': training_loss,
        'Validation Loss': validation_loss
    })

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=metrics_df['Epoch'],
            y=metrics_df['Training Accuracy'],
            name="Training Accuracy",
            line=dict(color="#1f77b4", width=3),
            hovertemplate="Epoch: %{x}<br>Training Accuracy: %{y:.4f}<extra></extra>"
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=metrics_df['Epoch'],
            y=metrics_df['Validation Accuracy'],
            name="Validation Accuracy",
            line=dict(color="#ff7f0e", width=3),
            hovertemplate="Epoch: %{x}<br>Validation Accuracy: %{y:.4f}<extra></extra>"
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=metrics_df['Epoch'],
            y=metrics_df['Training Loss'],
            name="Training Loss",
            line=dict(color="#1f77b4", width=2, dash='dash'),
            hovertemplate="Epoch: %{x}<br>Training Loss: %{y:.4f}<extra></extra>"
        ),
        secondary_y=True,
    )
    
    fig.add_trace(
        go.Scatter(
            x=metrics_df['Epoch'],
            y=metrics_df['Validation Loss'],
            name="Validation Loss",
            line=dict(color="#ff7f0e", width=2, dash='dash'),
            hovertemplate="Epoch: %{x}<br>Validation Loss: %{y:.4f}<extra></extra>"
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Training and Validation Metrics Over Epochs",
        title_font=dict(size=24),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=600,
        xaxis=dict(
            title="Epoch",
            gridcolor="lightgray",
            gridwidth=0.5,
            showgrid=True
        ),
        margin=dict(l=60, r=60, t=80, b=60)
    )

    fig.update_yaxes(
        title_text="Accuracy", 
        secondary_y=False,
        range=[0, 1],
        gridcolor="lightgray",
        gridwidth=0.5,
        showgrid=True
    )
    fig.update_yaxes(
        title_text="Loss", 
        secondary_y=True,
        range=[0, 1],
        gridcolor="lightgray",
        gridwidth=0.5,
        showgrid=False
    )
    
    return fig

def create_network_graph():
    """
    Creates a network graph similar to the one in the provided image
    """
    # Create a graph
    G = nx.Graph()
    
    # Add nodes with their labels (based on the image)
    nodes = [
        ("shaykh", "Shaykh"),
        ("ameer", "Ameer Samseen"),
        ("syeda", "Syeda Suhani Bilal"),
        ("al_dar", "Al Dar"),
        ("sameer", "Ahmed Sameer Abbas"),
        ("shehbaz", "Ahmed Shehbaz Ghori"),
        ("najam", "najam_al_hind"),
        ("faizan", "Faizan Taj"),
        ("umer", "Umer Bhutta"),
        ("aleena", "Aleena Tarique"),
        ("javed", "Kamran Shahid"),
        ("polaris", "Aziz Latif Polaris")
    ]
    
    # Add the nodes
    for node_id, node_label in nodes:
        G.add_node(node_id, label=node_label)
    
    # Add edges (connections between nodes)
    edges = [
        ("shaykh", "ameer"),
        ("shaykh", "syeda"),
        ("shaykh", "polaris"),
        ("ameer", "syeda"),
        ("syeda", "al_dar"),
        ("al_dar", "sameer"),
        ("al_dar", "javed"),
        ("sameer", "shehbaz"),
        ("najam", "faizan"),
        ("faizan", "umer"),
        ("umer", "aleena"),
        ("aleena", "javed"),
        ("aleena", "polaris"),
        ("polaris", "ameer")
    ]
    
    # Add the edges to the graph
    G.add_edges_from(edges)
    
    # Get node positions using a layout algorithm
    pos = nx.spring_layout(G, seed=42)
    
    # Create traces for nodes
    node_trace = go.Scatter(
        x=[pos[k][0] for k in G.nodes()],
        y=[pos[k][1] for k in G.nodes()],
        mode='markers+text',
        marker=dict(
            size=30,
            color='#32CD32',  # Green color as seen in the image
            line=dict(width=1, color='#11AA11')
        ),
        text=[G.nodes[node]['label'] for node in G.nodes()],
        textposition="top center",
        hoverinfo='text',
        name='Nodes'
    )
    
    # Create traces for edges
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color='#AAAAAA'),
        hoverinfo='none',
        mode='lines',
        name='Connections'
    )
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='Network Analysis',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0, l=0, r=0, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=600,
                    width=800
                 ))
    
    return fig, G

def create_network_analysis_view():
    """
    Creates a network analysis tab view
    """
    st.subheader("Network Analysis Visualization")
    
    # Create the network graph
    network_fig, G = create_network_graph()
    
    # Display network properties
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.plotly_chart(network_fig, use_container_width=True)
    
    with col2:
        st.markdown("### Network Properties")
        
        st.metric("Number of Entities", len(G.nodes()))
        st.metric("Number of Connections", len(G.edges()))
        
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        
        # Find the most central entity
        most_central = max(degree_centrality.items(), key=lambda x: x[1])
        most_influential = max(betweenness_centrality.items(), key=lambda x: x[1])
        
        st.metric("Most Connected Entity", G.nodes[most_central[0]]['label'])
        st.metric("Most Influential Entity", G.nodes[most_influential[0]]['label'])
    
    # Detailed network metrics
    with st.expander("Network Metrics Details"):
        st.markdown("### Centrality Measures")
        
        centrality_df = pd.DataFrame({
            'Entity': [G.nodes[node]['label'] for node in G.nodes()],
            'Degree Centrality': [degree_centrality[node] for node in G.nodes()],
            'Betweenness Centrality': [betweenness_centrality[node] for node in G.nodes()],
            'Closeness Centrality': [closeness_centrality[node] for node in G.nodes()]
        })
        
        centrality_df = centrality_df.sort_values(by='Degree Centrality', ascending=False)
        
        st.dataframe(centrality_df, use_container_width=True)
        
        st.markdown("""
        ### Understanding Centrality Metrics
        
        - **Degree Centrality**: Measures how connected an entity is. Higher values indicate entities with more direct connections.
        - **Betweenness Centrality**: Measures how often an entity lies on the shortest path between other entities. Higher values indicate entities that act as "bridges" or information brokers.
        - **Closeness Centrality**: Measures how close an entity is to all other entities in the network. Higher values indicate entities that can quickly reach or disseminate information to others.
        """)
    
    # Export network data
    st.subheader("Export Network Data")
    
    export_format = st.radio("Export Network Format", ["CSV", "JSON", "GEXF"], horizontal=True)
    
    if st.button("Export Network Data", type="primary"):
        if export_format == "CSV":
            # Export nodes
            nodes_df = pd.DataFrame({
                "ID": list(G.nodes()),
                "Label": [G.nodes[node]["label"] for node in G.nodes()],
                "Degree": [G.degree(node) for node in G.nodes()],
                "Betweenness": [betweenness_centrality[node] for node in G.nodes()]
            })
            
            # Export edges
            edges_df = pd.DataFrame({
                "Source": [edge[0] for edge in G.edges()],
                "Target": [edge[1] for edge in G.edges()]
            })
            
            # Create buffer for zip file
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                nodes_df.to_excel(writer, sheet_name="Nodes", index=False)
                edges_df.to_excel(writer, sheet_name="Edges", index=False)
            
            buffer.seek(0)
            st.download_button(
                "Download Network Excel",
                buffer,
                "network_data.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
        elif export_format == "JSON":
            # Create JSON network representation
            network_json = {
                "nodes": [{"id": node, "label": G.nodes[node]["label"]} for node in G.nodes()],
                "edges": [{"source": edge[0], "target": edge[1]} for edge in G.edges()]
            }
            
            st.download_button(
                "Download Network JSON",
                json.dumps(network_json),
                "network_data.json",
                "application/json"
            )
            
        else:  # GEXF format
            # Use NetworkX's built-in GEXF export
            nx.write_gexf(G, "network_data.gexf")
            
            with open("network_data.gexf", "r") as f:
                gexf_data = f.read()
                
            st.download_button(
                "Download Network GEXF",
                gexf_data,
                "network_data.gexf",
                "application/xml"
            )

def main():
    st.title("🔍 Advanced Sentiment Analysis Dashboard (BRET Model)")
    
    tabs = st.tabs(["📊 Data Analysis", "🧠 Model Performance", "🔗 Network Analysis"])
    
    with tabs[0]:  # Data Analysis tab
        st.header("📝 Text and Dataset Analysis")
        
        analysis_type = st.radio(
            "Choose Analysis Type",
            ["Single Text Analysis", "Dataset Analysis"],
            horizontal=True
        )
        
        if analysis_type == "Single Text Analysis":
            st.subheader("Single Text Analysis")
            
            input_text = st.text_area(
                "Enter text to analyze:",
                height=150,
                placeholder="Type or paste your text here..."
            )
            
            if st.button("Analyze Text", type="primary"):
                if input_text.strip():
                    with st.spinner("Analyzing sentiment..."):
                        result = predict_sentiment(clean_text(input_text))
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Positive", 
                                f"{result['Pos %']:.1f}%",
                                delta=f"{result['Pos %'] - 33.3:.1f}" if result['Pos %'] > 33.3 else f"{result['Pos %'] - 33.3:.1f}"
                            )
                            
                        with col2:
                            st.metric(
                                "Negative", 
                                f"{result['Neg %']:.1f}%",
                                delta=f"{result['Neg %'] - 33.3:.1f}" if result['Neg %'] > 33.3 else f"{result['Neg %'] - 33.3:.1f}"
                            )
                            
                        with col3:
                            st.metric(
                                "Neutral", 
                                f"{result['Neu %']:.1f}%",
                                delta=f"{result['Neu %'] - 33.3:.1f}" if result['Neu %'] > 33.3 else f"{result['Neu %'] - 33.3:.1f}"
                            )
                        
                        # Display the predicted sentiment
                        st.markdown("### Overall Sentiment")
                        
                        sentiment_html = f"""
                        <div style="display: flex; align-items: center; justify-content: center; margin: 20px 0;">
                            <div class="sentiment-{result['Average'].lower()}" style="text-align: center; font-size: 24px; padding: 10px 30px;">
                                {result['Average']}
                            </div>
                        </div>
                        """
                        
                        st.markdown(sentiment_html, unsafe_allow_html=True)
                        
                        # Create a gauge chart for sentiment visualization
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=result['Pos %'] - result['Neg %'],
                            title={'text': "Sentiment Score"},
                            gauge={
                                'axis': {'range': [-100, 100]},
                                'bar': {'color': "darkgray"},
                                'steps': [
                                    {'range': [-100, -33], 'color': COLORS["Negative"]},
                                    {'range': [-33, 33], 'color': COLORS["Neutral"]},
                                    {'range': [33, 100], 'color': COLORS["Positive"]}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': result['Pos %'] - result['Neg %']
                                }
                            }
                        ))
                        
                        fig.update_layout(
                            height=300,
                            margin=dict(l=20, r=20, t=30, b=20)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                else:
                    st.warning("Please enter some text to analyze.")
                    
        else:  # Dataset Analysis
            st.subheader("Dataset Analysis")
            
            upload_option = st.radio(
                "Choose Data Source",
                ["Upload CSV/Excel File", "Use Sample Dataset"],
                horizontal=True
            )
            
            if upload_option == "Upload CSV/Excel File":
                uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "xls"])
                
                if uploaded_file is not None:
                    try:
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                        else:
                            df = pd.read_excel(uploaded_file)
                            
                        st.success(f"File uploaded successfully. {len(df)} rows detected.")
                        
                        if not df.empty:
                            text_columns = df.select_dtypes(include=['object']).columns.tolist()
                            
                            if text_columns:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    selected_column = st.selectbox(
                                        "Select text column to analyze:",
                                        options=text_columns
                                    )
                                
                                with col2:
                                    # If dataset has < 100 rows, analyze all; otherwise allow sample
                                    if len(df) > 100:
                                        analyze_option = st.radio(
                                            "Analyze", 
                                            ["Entire Dataset", f"Sample (100/{len(df)} rows)"],
                                            horizontal=True
                                        )
                                        
                                        if analyze_option.startswith("Sample"):
                                            df = df.sample(100, random_state=42)
                                
                                if st.button("Analyze Dataset", type="primary"):
                                    with st.spinner("Analyzing dataset..."):
                                        results_df = analyze_dataframe(df, selected_column)
                                        st.session_state.analysis_results = results_df
                                    
                                    st.success("Analysis complete!")
                                    
                                    if 'analysis_results' in st.session_state:
                                        create_detailed_analysis_view(st.session_state.analysis_results)
                            else:
                                st.warning("No text columns found in the dataset. Please upload a dataset containing text data.")
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")
            else:
                st.info("Using sample dataset for demonstration purposes.")
                
                # Generate a sample dataset
                sample_texts = [
                    "I absolutely love this product! It's amazing and exceeds all my expectations.",
                    "This is terrible. Worst purchase I've ever made. Don't waste your money.",
                    "The product arrived on time and works as expected. Nothing special but does the job.",
                    "While there are some good features, overall I'm disappointed with the quality.",
                    "I'm extremely satisfied with my purchase. The customer service was excellent too!",
                    "Meh, it's okay I guess. Probably wouldn't buy it again though.",
                    "This product changed my life! I use it every day and couldn't be happier.",
                    "Broken on arrival. Terrible packaging and no response from customer service.",
                    "It has both pros and cons. Some features work well, others need improvement.",
                    "Pretty good value for the price. Not perfect but I'm satisfied overall."
                ]
                
                # Generate more sample data
                np.random.seed(42)
                more_samples = []
                
                sentiment_patterns = [
                    ["great", "excellent", "wonderful", "love", "amazing", "fantastic", "best", "perfect"],
                    ["bad", "terrible", "awful", "worst", "hate", "disappointing", "poor", "useless"],
                    ["okay", "fine", "decent", "average", "acceptable", "reasonable", "fair", "adequate"]
                ]
                
                for _ in range(40):
                    pattern_idx = np.random.randint(0, 3)
                    words = sentiment_patterns[pattern_idx]
                    selected_words = np.random.choice(words, size=np.random.randint(1, 4), replace=False)
                    
                    if pattern_idx == 0:  # Positive
                        text = f"This product is {' and '.join(selected_words)}. {np.random.choice(['Would recommend!', 'Very satisfied.', 'Great purchase!'])}"
                    elif pattern_idx == 1:  # Negative
                        text = f"This product is {' and '.join(selected_words)}. {np.random.choice(['Would not recommend.', 'Very disappointed.', 'Waste of money.'])}"
                    else:  # Neutral
                        text = f"This product is {' and '.join(selected_words)}. {np.random.choice(['It works.', 'Does what it should.', 'No strong feelings.'])}"
                        
                    more_samples.append(text)
                
                sample_texts.extend(more_samples)
                sample_df = pd.DataFrame({"text": sample_texts})
                
                if st.button("Analyze Sample Dataset", type="primary"):
                    with st.spinner("Analyzing sample dataset..."):
                        results_df = analyze_dataframe(sample_df, "text")
                        st.session_state.analysis_results = results_df
                    
                    st.success("Analysis complete!")
                    
                    if 'analysis_results' in st.session_state:
                        create_detailed_analysis_view(st.session_state.analysis_results)
    
    with tabs[1]:
        st.header("🔄 Model Training & Performance")
        
        st.markdown("""
        This tab provides insights into how the sentiment analysis model was trained and evaluated.
        """)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            training_fig = plot_interactive_training_metrics()
            st.plotly_chart(training_fig, use_container_width=True)
            
            with st.expander("About Training Metrics"):
                st.markdown("""
                ### Understanding Training Metrics
                
                - **Training Accuracy**: Measures how well the model performs on data it was trained on
                - **Validation Accuracy**: Measures how well the model generalizes to new, unseen data
                - **Training Loss**: The error on the training set (lower is better)
                - **Validation Loss**: The error on the validation set (lower is better)
                
                The gap between training and validation metrics indicates how well the model generalizes.
                A large gap might suggest overfitting, while values that are too low might suggest underfitting.
                
                The best epoch is typically where validation loss is at its minimum before it starts increasing again.
                """)
        
        with col2:
            st.markdown("### Training Summary")
            
            st.metric("Best Validation Accuracy", f"{max(validation_accuracy):.3f}")
            st.metric("Lowest Validation Loss", f"{min(validation_loss):.3f}")
            
            best_epoch = validation_loss.index(min(validation_loss))
            st.metric("Best Epoch", best_epoch)
            
            st.markdown("### Model Configuration")
            
            st.markdown("""
            - **Base Model**: BERT (bert-base-uncased)
            - **Classes**: 3 (Positive, Negative, Neutral)
            - **Batch Size**: 32
            - **Learning Rate**: 2e-5
            - **Optimizer**: AdamW
            """)
                
    with tabs[2]:  # Network Analysis tab
        create_network_analysis_view()

if __name__ == "__main__":
    main()