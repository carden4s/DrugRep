# drug_repurposing_explorer.py
import streamlit as st
import networkx as nx
from node2vec import Node2Vec
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl

# Professional color scheme
NEUTRAL_BG = "#f8f9fa"
PRIMARY_COLOR = "#2c3e50"
SECONDARY_COLOR = "#3498db"
ACCENT_COLOR = "#2980b9"
NODE_COLORS = {
    "drug": "#3498db",      # Professional blue
    "protein": "#27ae60",   # Professional green
    "disease": "#8e44ad"    # Professional purple
}

# Set professional matplotlib style
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['grid.color'] = '#e0e0e0'
mpl.rcParams['font.family'] = 'sans-serif'

# Page configuration
st.set_page_config(
    page_title="Drug Repurposing Explorer",
    layout="centered",
    page_icon="💊",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown(f"""
<style>
    .stApp {{
        background-color: {NEUTRAL_BG};
    }}
    .stMarkdown {{
        color: {PRIMARY_COLOR};
    }}
    .st-bq {{
        border-left-color: {ACCENT_COLOR} !important;
    }}
    .stAlert {{
        border-left: 4px solid {ACCENT_COLOR} !important;
    }}
    .header {{
        color: {PRIMARY_COLOR};
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }}
    .metric-box {{
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }}
    .prediction-container {{
        background-color: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        margin: 1.5rem 0;
    }}
</style>
""", unsafe_allow_html=True)

# Title with professional styling
st.markdown(f"<h1 style='color:{PRIMARY_COLOR};'>Drug-Disease Link Prediction</h1>", unsafe_allow_html=True)
st.markdown("""
<p style="color:#7f8c8d; margin-top:-10px;">
A computational approach to identify potential therapeutic relationships using graph-based machine learning
</p>
""", unsafe_allow_html=True)

# -- Step 1: Knowledge Graph Construction --
st.markdown("<h2 class='header'>Biological Knowledge Graph</h2>", unsafe_allow_html=True)
st.markdown("""
<p style="color:#555;">
A structured representation of relationships between pharmaceutical compounds, 
protein targets, and disease states based on established biomedical knowledge.
</p>
""", unsafe_allow_html=True)

@st.cache_data
def build_graph():
    G = nx.Graph()
    nodes = {
        "drug": ["Aspirin", "Ibuprofen", "Metformin", "Atorvastatin", "Simvastatin"],
        "protein": ["COX1", "AMPK", "HMGCR", "IL6", "TNFa", "PPARg", "ACE2"],
        "disease": ["Inflammation", "Type 2 Diabetes", "Headache", "Cardiovascular Disease", "Alzheimer's Disease"]
    }
    
    # Add nodes with metadata
    for node_type, node_list in nodes.items():
        for node in node_list:
            G.add_node(node, type=node_type)
    
    # Add relationships with biological explanations
    edges = [
        # Drug-protein relationships
        ("Aspirin", "COX1", {"desc": "COX-1 enzyme inhibition"}),
        ("Ibuprofen", "COX1", {"desc": "COX-1 enzyme inhibition"}),
        ("Metformin", "AMPK", {"desc": "AMPK activation"}),
        ("Atorvastatin", "HMGCR", {"desc": "HMG-CoA reductase inhibition"}),
        ("Simvastatin", "HMGCR", {"desc": "HMG-CoA reductase inhibition"}),
        
        # Protein-disease relationships
        ("COX1", "Inflammation", {"desc": "Prostaglandin-mediated inflammatory response"}),
        ("AMPK", "Type 2 Diabetes", {"desc": "Glucose metabolism regulation"}),
        ("HMGCR", "Cardiovascular Disease", {"desc": "Cholesterol biosynthesis pathway"}),
        ("IL6", "Inflammation", {"desc": "Pro-inflammatory cytokine signaling"}),
        ("TNFa", "Inflammation", {"desc": "Pro-inflammatory cytokine signaling"}),
        ("PPARg", "Type 2 Diabetes", {"desc": "Insulin sensitization pathway"}),
        ("ACE2", "Cardiovascular Disease", {"desc": "Renin-angiotensin system regulation"}),
        
        # Drug-disease relationships (known indications)
        ("Aspirin", "Headache", {"desc": "Analgesic effect"}),
        ("Ibuprofen", "Headache", {"desc": "Analgesic and anti-inflammatory effect"}),
        ("Atorvastatin", "Cardiovascular Disease", {"desc": "LDL cholesterol reduction"}),
        ("Metformin", "Type 2 Diabetes", {"desc": "First-line glycemic control"}),
        ("Simvastatin", "Cardiovascular Disease", {"desc": "LDL cholesterol reduction"})
    ]
    
    # Add edges with attributes
    for edge in edges:
        G.add_edge(edge[0], edge[1], desc=edge[2]["desc"])
    
    return G

G = build_graph()
drug_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'drug']
disease_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'disease']

# Graph visualization
fig, ax = plt.subplots(figsize=(9, 7))
layout = nx.spring_layout(G, seed=42, k=0.6)

# Draw nodes by type
for node_type, color in NODE_COLORS.items():
    nodes = [n for n, d in G.nodes(data=True) if d['type'] == node_type]
    nx.draw_networkx_nodes(
        G, layout,
        nodelist=nodes,
        node_color=color,
        edgecolors='#2c3e50',
        node_size=1200,
        ax=ax
    )

# Draw edges
nx.draw_networkx_edges(G, layout, width=1.2, alpha=0.6, edge_color='#95a5a6', ax=ax)

# Draw labels
nx.draw_networkx_labels(G, layout, font_size=9, font_weight='normal', font_color='#2c3e50', ax=ax)

# Add legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=NODE_COLORS['drug'], 
               markersize=10, label='Drugs', markeredgecolor='#2c3e50'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=NODE_COLORS['protein'], 
               markersize=10, label='Proteins', markeredgecolor='#2c3e50'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=NODE_COLORS['disease'], 
               markersize=10, label='Diseases', markeredgecolor='#2c3e50')
]
ax.legend(handles=legend_elements, loc='best', frameon=True, framealpha=0.9)

ax.set_title("Drug-Protein-Disease Interaction Network", fontsize=14)
ax.set_facecolor('white')
plt.axis('off')
st.pyplot(fig)

# Biological relationships explanation
with st.expander("Biological Relationships Documentation"):
    st.markdown("""
    **Established biological relationships in the knowledge graph:**
    
    | Relationship | Biological Mechanism |
    |--------------|----------------------|
    | Aspirin/Ibuprofen - COX1 | Non-steroidal anti-inflammatory drugs (NSAIDs) inhibit cyclooxygenase-1 enzyme |
    | Metformin - AMPK | Activates AMP-activated protein kinase to improve insulin sensitivity |
    | Atorvastatin/Simvastatin - HMGCR | Statins inhibit HMG-CoA reductase, a rate-limiting enzyme in cholesterol biosynthesis |
    | COX1 - Inflammation | Mediates production of pro-inflammatory prostaglandins |
    | AMPK - Type 2 Diabetes | Regulates glucose uptake and metabolism |
    | HMGCR - Cardiovascular Disease | Elevated LDL cholesterol is a major risk factor for cardiovascular disease |
    | IL6/TNFa - Inflammation | Pro-inflammatory cytokines involved in acute phase response |
    | PPARg - Type 2 Diabetes | Nuclear receptor that regulates adipocyte differentiation and glucose metabolism |
    | ACE2 - Cardiovascular Disease | Key enzyme in the renin-angiotensin-aldosterone system (RAAS) |
    """)

# -- Step 2: Computational Methodology --
st.markdown("<h2 class='header'>Computational Methodology</h2>", unsafe_allow_html=True)
st.markdown("""
<p style="color:#555;">
The analytical pipeline employs graph representation learning and machine learning 
to identify potential therapeutic relationships based on network proximity.
</p>
""", unsafe_allow_html=True)

# Embeddings generation
st.markdown("<h3 style='color:#2c3e50;'>Node Embedding Generation</h3>", unsafe_allow_html=True)
st.markdown("""
<p style="color:#555;">
Node2Vec algorithm learns vector representations of biological entities by 
simulating random walks through the knowledge graph, capturing structural relationships.
</p>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def generate_embeddings(_graph):
    n2v = Node2Vec(
        _graph, 
        dimensions=64,
        walk_length=30, 
        num_walks=200,
        workers=2,
        quiet=True
    )
    model = n2v.fit(window=10, min_count=1, batch_words=4)
    return {node: model.wv[node] for node in _graph.nodes()}

with st.spinner("Generating node embeddings (this may take 15-25 seconds)..."):
    embeddings = generate_embeddings(G)
st.success("Node embeddings successfully generated")

# Predictive model training
st.markdown("<h3 style='color:#2c3e50;'>Predictive Model Training</h3>", unsafe_allow_html=True)
st.markdown("""
<p style="color:#555;">
A logistic regression classifier is trained to predict therapeutic relationships 
using concatenated drug-disease embedding vectors as input features.
</p>
""", unsafe_allow_html=True)

@st.cache_data
def train_classifier(emb):
    # Known therapeutic relationships
    positives = [
        ("Aspirin", "Headache"),
        ("Ibuprofen", "Headache"),
        ("Atorvastatin", "Cardiovascular Disease"),
        ("Simvastatin", "Cardiovascular Disease"),
        ("Metformin", "Type 2 Diabetes")
    ]
    
    # Non-therapeutic relationships
    negatives = [
        ("Metformin", "Headache"),
        ("Aspirin", "Type 2 Diabetes"),
        ("Ibuprofen", "Cardiovascular Disease"),
        ("Atorvastatin", "Headache"),
        ("Simvastatin", "Inflammation"),
        ("Metformin", "Cardiovascular Disease"),
        ("Aspirin", "Cardiovascular Disease")
    ]
    
    # Create training dataset
    X = []
    y = []
    for drug, disease in positives:
        X.append(np.hstack([emb[drug], emb[disease]]))
        y.append(1)
    
    for drug, disease in negatives:
        X.append(np.hstack([emb[drug], emb[disease]]))
        y.append(0)
    
    # Train model
    model = LogisticRegression(max_iter=2000, class_weight='balanced').fit(X, y)
    return model, X, y, positives, negatives

model, X_train, y_train, positives, negatives = train_classifier(embeddings)
accuracy = model.score(X_train, y_train)

# Model performance metrics
st.markdown(f"""
<div class="metric-box">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h4 style="margin: 0 0 5px 0; color: #2c3e50;">Model Performance</h4>
            <p style="margin: 0; color: #7f8c8d; font-size: 0.9rem;">
                Training accuracy: <strong>{accuracy:.1%}</strong>
            </p>
            <p style="margin: 5px 0 0 0; color: #7f8c8d; font-size: 0.9rem;">
                Training samples: {len(X_train)} ({len(positives)} positive, {len(negatives)} negative)
            </p>
        </div>
        <div style="font-size: 2rem; color: #3498db;">
            {accuracy:.0%}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -- Step 3: Predictive Analysis --
st.markdown("<h2 class='header'>Predictive Analysis</h2>", unsafe_allow_html=True)

st.markdown("""
<p style="color:#555;">
Evaluate potential therapeutic relationships between pharmaceutical compounds and disease states.
</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    selected_drug = st.selectbox("Select a pharmaceutical compound:", drug_nodes, index=0)
with col2:
    selected_disease = st.selectbox("Select a disease state:", disease_nodes, index=0)

# Create input vector and make prediction
input_vector = np.hstack([embeddings[selected_drug], embeddings[selected_disease]])
probability = model.predict_proba([input_vector])[0][1]

# Determine prediction confidence level
if probability > 0.75:
    confidence = "High"
    color = "#27ae60"
    interpretation = "Strong predicted therapeutic relationship"
elif probability > 0.5:
    confidence = "Moderate"
    color = "#f39c12"
    interpretation = "Potential therapeutic relationship"
else:
    confidence = "Low"
    color = "#e74c3c"
    interpretation = "Low probability of therapeutic relationship"

# Prediction result display
st.markdown(f"""
<div style="text-align: center;">
    <h3 style="color: #2c3e50; margin-top: 0;">Prediction Result</h3>

    <div style="display: flex; justify-content: center; margin-bottom: 1rem;">
        <div>
            <div style="font-size: 2.5rem; font-weight: bold; color: {color};">
                {probability:.2f}
            </div>
            <div style="font-size: 1rem; color: #7f8c8d; margin-top: 0.5rem;">
                Probability Score
            </div>
        </div>
    </div>

    <div style="margin-top: 1rem; padding: 1rem; background-color: #f8f9fa; border-radius: 8px;">
        <p style="margin: 0; font-size: 1.1rem; color: #2c3e50;">
            <strong>{selected_drug}</strong> → <strong>{selected_disease}</strong>
        </p>
        <p style="margin: 0.5rem 0 0 0; font-size: 1rem; color: {color};">
            {interpretation} ({confidence} confidence)
        </p>
    </div>
</div>
""", unsafe_allow_html=True)


# -- Step 4: Embedding Space Visualization --
st.markdown("<h2 class='header'>Embedding Space Analysis</h2>", unsafe_allow_html=True)
st.markdown("""
<p style="color:#555;">
Principal Component Analysis (PCA) projection of 64-dimensional node embeddings 
into 2-dimensional space for visualization.
</p>
""", unsafe_allow_html=True)

# Compute PCA
all_nodes = list(embeddings.keys())
matrix = np.vstack([embeddings[node] for node in all_nodes])
pca = PCA(n_components=2).fit_transform(matrix)

# Create figure
fig2, ax2 = plt.subplots(figsize=(8, 6))

# Plot points with type-based coloring
for node in all_nodes:
    idx = all_nodes.index(node)
    node_type = G.nodes[node]['type']
    ax2.scatter(
        pca[idx, 0], pca[idx, 1], 
        color=NODE_COLORS[node_type], 
        s=120,
        alpha=0.8,
        edgecolor='#2c3e50',
        linewidth=0.5
    )
    ax2.annotate(node, (pca[idx, 0], pca[idx, 1]), 
                fontsize=8, 
                xytext=(3, 3), 
                textcoords='offset points')

# Chart configuration
ax2.set_title("PCA Projection of Node Embeddings", fontsize=13)
ax2.set_xlabel("Principal Component 1", fontsize=10)
ax2.set_ylabel("Principal Component 2", fontsize=10)
ax2.grid(True, alpha=0.2)
ax2.set_facecolor('white')

# Create legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=NODE_COLORS['drug'], 
               markersize=8, label='Drugs', markeredgecolor='#2c3e50'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=NODE_COLORS['protein'], 
               markersize=8, label='Proteins', markeredgecolor='#2c3e50'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=NODE_COLORS['disease'], 
               markersize=8, label='Diseases', markeredgecolor='#2c3e50')
]
ax2.legend(handles=legend_elements, loc='best', frameon=True, framealpha=0.9)

st.pyplot(fig2)

# -- Step 5: Methodology Overview --
st.markdown("<h2 class='header'>Methodology Overview</h2>", unsafe_allow_html=True)
st.markdown("""
<div style="background-color: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
    <h3 style="color: #2c3e50; margin-top: 0;">Computational Framework</h3>
    
    <p style="color: #555;">
    This analytical pipeline employs graph-based machine learning to identify potential 
    drug repurposing opportunities through the following steps:
    </p>
    
    <ol style="color: #555;">
        <li><strong>Knowledge Graph Construction</strong>: Integration of pharmaceutical, 
        proteomic, and disease entities with established biological relationships</li>
        <li><strong>Node Embedding Generation</strong>: Representation learning using Node2Vec 
        to capture structural relationships in a continuous vector space</li>
        <li><strong>Predictive Model Development</strong>: Logistic regression classifier 
        trained on known therapeutic relationships</li>
        <li><strong>Therapeutic Relationship Prediction</strong>: Probabilistic assessment 
        of novel drug-disease relationships</li>
    </ol>
    
    <h3 style="color: #2c3e50;">Analytical Validation</h3>
    <p style="color: #555;">
    The predictive model achieved a training accuracy of <strong>{accuracy:.1%}</strong> 
    using {len(X_train)} samples ({len(positives)} known therapeutic relationships, 
    {len(negatives)} negative controls).
    </p>
    
    <h3 style="color: #2c3e50;">Applications</h3>
    <p style="color: #555;">
    This computational approach enables systematic identification of:
    </p>
    <ul style="color: #555;">
        <li>Novel therapeutic indications for existing pharmaceutical compounds</li>
        <li>Potential mechanisms of action through protein interaction networks</li>
        <li>Candidate compounds for further experimental validation</li>
    </ul>
</div>
""".format(accuracy=accuracy, len_X_train=len(X_train), len_positives=len(positives), len_negatives=len(negatives)), 
unsafe_allow_html=True)

# Real-world example
st.markdown("""
<div style="margin-top: 1.5rem; background-color: #e3f2fd; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #1976d2;">
    <h3 style="color: #0d47a1; margin-top: 0;">Clinical Validation: Metformin</h3>
    <p style="color: #37474f;">
    Originally developed for type 2 diabetes, metformin has demonstrated potential therapeutic 
    benefits in cancer prevention and treatment through AMPK-mediated pathways. This repurposing 
    success highlights the value of computational approaches in identifying novel therapeutic applications.
    </p>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; font-size: 0.9rem; padding: 1rem 0;">
    <p>Drug Repurposing Explorer v2.1 | Computational Pharmacology Platform</p>
    <p>© 2023 Biomedical Informatics Research Group</p>
</div>
""", unsafe_allow_html=True)