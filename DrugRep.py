# drug_repurposing_explorer.py
import streamlit as st
import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.graph_objects as go
from scipy.sparse import lil_matrix, csr_matrix
from collections import defaultdict
import random
import sys
import shap

# Page configuration must be first Streamlit command
st.set_page_config(
    page_title="Drug Repurposing Explorer",
    layout="centered",
    page_icon="💊",
    initial_sidebar_state="expanded"
)

# Check Python version
st.sidebar.markdown(f"**Python Version:** `{sys.version}`")
st.sidebar.markdown(f"**Streamlit Version:** `{st.__version__}`")

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
        border-radius: 8极;
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
    .methodology-card {{
        background-color: white;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid {SECONDARY_COLOR};
    }}
    .step-card {{
        background-color: #f8f9fa;
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 3px solid {ACCENT_COLOR};
    }}
    .shap-container {{
        background-color: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-top: 1.5rem;
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

# ========================
# MODULAR FUNCTIONS
# ========================

@st.cache_data
def build_graph():
    """Build the biological knowledge graph with drugs, proteins, and diseases"""
    G = nx.Graph()
    nodes = {
        "drug": ["Aspirin", "Ibuprofen", "Metformin", "Atorvastatin", "Simvastatin"],
        "protein": ["COX1", "AMPK", "HMGCR", "IL6", "TNFa", "PPARg", "ACE2"],
        "disease": ["Inflammation", "Type 2 Diabetes", "Headache", 
                    "Cardiovascular Disease", "Alzheimer's Disease"]
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
        ("Metformin", "AMPK", {"desc极": "AMPK activation"}),
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

@st.cache_data(show_spinner=False)
def generate_embeddings(
    _graph: nx.Graph,
    num_walks: int = 100,
    walk_length: int = 20,
    window_size: int = 4,
    max_embedding_dim: int = 32,
    random_seed: int = 42
) -> tuple[dict, int]:
    """Generate node embeddings using random walks and PPMI matrix factorization"""
    random.seed(random_seed)
    nodes = list(_graph.nodes())
    node_to_index = {node: i for i, node in enumerate(nodes)}
    N = len(nodes)
    
    # Dynamically adjust embedding dimension
    embedding_dim = min(max_embedding_dim, N - 1)  # Ensure it's <= n_features
    if embedding_dim < max_embedding_dim:
        st.warning(f"Reduced embedding dimension from {max_embedding_dim} to {embedding_dim} to match graph size")

    # 1) Random walks
    walks = []
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            current = node
            for _ in range(walk_length - 1):
                neighbors = list(_graph.neighbors(current))
                if not neighbors:
                    break
                current = random.choice(neighbors)
                walk.append(current)
            walks.append([node_to_index[n] for n in walk])

    # 2) Build sparse co-occurrence matrix
    cooc = lil_matrix((N, N), dtype=np.float32)
    for walk in walks:
        for pos, node_idx in enumerate(walk):
            start = max(0, pos - window_size)
            end = min(len(walk), pos + window_size + 1)
            for ctx_pos in range(start, end):
                if pos == ctx_pos:
                    continue
                ctx_idx = walk[ctx_pos]
                # Weight by distance
                cooc[node_idx, ctx_idx] += 1.0 / (abs(pos - ctx_pos) + 1)

    # 3) Convert to CSR for efficiency
    cooc = cooc.tocsr()
    total = cooc.sum()
    row_sums = np.array(cooc.sum(axis=1)).ravel() + 1e-8
    col_sums = np.array(cooc.sum(axis=0)).ravel() + 1e-8

    # 4) Compute PPMI
    ppmi = lil_matrix((N, N), dtype=np.float32)
    rows, cols = cooc.nonzero()
    for i, j in zip(rows, cols):
        count = cooc[i, j]
        if count > 0:
            pmi = np.log((count * total) / (row_sums[i] * col_sums[j]))
            ppmi[i, j] = max(0, pmi)
    
    ppmi = ppmi.tocsr()

    # 5) Truncated SVD
    svd = TruncatedSVD(n_components=embedding_dim, random_state=random_seed)
    embeddings_mat = svd.fit_transform(ppmi)

    return {node: embeddings_mat[node_to_index[node]] for node in nodes}, embedding_dim

@st.cache_data
def train_classifier(emb, test_size=0.2):
    """Train and evaluate a classifier with proper validation"""
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
        ("Aspirin", "Cardiovascular Disease"),
        ("Ibuprofen", "Alzheimer's Disease"),
        ("Metformin", "Alzheimer's Disease"),
        ("Simvastatin", "Type 2 Diabetes"),
        ("Atorvastatin", "Alzheimer's Disease")
    ]
    
    # Create dataset
    X = []
    y = []
    for drug, disease in positives:
        X.append(np.hstack([emb[drug], emb[disease]]))
        y.append(1)
    
    for drug, disease in negatives:
        X.append(np.hstack([emb[drug], emb[disease]]))
        y.append(0)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    # Train model
    model = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=5, scoring='accuracy'
    )
    cv_accuracy = np.mean(cv_scores)
    
    # Test set evaluation
    y_pred = model.predict(X_test)
    test_accuracy = model.score(X_test, y_test)
    report = classification_report(y_test, y_pred, target_names=["Negative", "Positive"])
    
    return model, X, y, positives, negatives, cv_accuracy, test_accuracy, report

def generate_shap_explanation(model, sample, feature_names=None):
    """Generate SHAP explanation for a prediction"""
    explainer = shap.Explainer(model)
    shap_values = explainer(sample)
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0], max_display=15, show=False)
    plt.tight_layout()
    return fig

# ========================
# MAIN APPLICATION
# ========================

# -- Step 1: Knowledge Graph Construction --
st.markdown("<h2 class='header'>Biological Knowledge Graph</h2>", unsafe_allow_html=True)
st.markdown("""
<p style="color:#555;">
A structured representation of relationships between pharmaceutical compounds, 
protein targets, and disease states based on established biomedical knowledge.
</p>
""", unsafe_allow_html=True)

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

# Draw labels with improved positioning
for node, pos in layout.items():
    node_type = G.nodes[node]['type']
    offset = (0, 0)
    if node_type == 'drug':
        offset = (0, -0.02)
    elif node_type == 'disease':
        offset = (0, 0.02)
    ax.text(pos[0] + offset[0], pos[1] + offset[1], node, 
            fontsize=9, ha='center', va='center', 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

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
We generate vector representations of biological entities using random walks 
through the knowledge graph, capturing structural relationships.
</p>
""", unsafe_allow_html=True)

# Scalability note
with st.expander("Scalability Considerations"):
    st.markdown("""
    **For large-scale applications:**
    
    - This demo uses a simplified embedding method suitable for small graphs
    - For production use with larger knowledge graphs:
        - Use optimized libraries like Node2Vec or DeepWalk
        - Consider GPU-accelerated implementations
        - Employ distributed computing frameworks
        - Use incremental learning approaches
    - Current parameters optimized for demonstration purposes
    """)

with st.spinner("Generating node embeddings (this may take 10-15 seconds)..."):
    embeddings, actual_dim = generate_embeddings(G)
st.success(f"Node embeddings successfully generated ({actual_dim}-dimensional)")

# Predictive model training
st.markdown("<h3 style='color:#2c3e50;'>Predictive Model Training</h3>", unsafe_allow_html=True)
st.markdown("""
<p style="color:#555;">
A logistic regression classifier is trained to predict therapeutic relationships 
using concatenated drug-disease embedding vectors as input features.
</p>
""", unsafe_allow_html=True)

model, X, y, positives, negatives, cv_accuracy, test_accuracy, report = train_classifier(embeddings)

# Class balance information
st.markdown(f"""
<div class="metric-box">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h4 style="margin: 0 0 5px 0; color: #2c3e50;">Class Distribution</h4>
            <p style="margin: 0; color: #7f8c8d; font-size: 0.9rem;">
                Positive samples: <strong>{len(positives)}</strong> (known therapeutic relationships)
            </p>
            <p style="margin: 5px 0 0 0; color: #7f8c8d; font-size: 0.9rem;">
                Negative samples: <strong>{len(negatives)}</strong> (non-therapeutic relationships)
            </p>
        </div>
        <div style="font-size: 2rem; color: #8e44ad;">
            {len(positives)}:{len(negatives)}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Model performance metrics
st.markdown(f"""
<div class="metric-box">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h4 style="margin: 0 0 5px 0; color: #2c3e50;">Model Performance</h4>
            <p style="margin: 0; color: #7f8c8d; font-size: 0.9rem;">
                Cross-validation accuracy: <strong>{cv_accuracy:.1%}</strong>
            </p>
            <p style="margin: 5px 0 0 0; color: #7f8c8d; font-size: 0.9rem;">
                Test set accuracy: <strong>{test_accuracy:.1%}</strong>
            </p>
        </div>
        <div style="font-size: 2rem; color: #3498db;">
            {test_accuracy:.0%}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Detailed classification report
with st.expander("Detailed Classification Report"):
    st.text(report)

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

# Enhanced Prediction Result with Plotly gauge
st.markdown("<h3 style='color: #2c3e50; margin-top: 1rem;'>Prediction Result</h3>", unsafe_allow_html=True)

# Create Plotly gauge chart
fig_gauge = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = probability,
    number = {'suffix': " probability", 'font': {'size': 24}},
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': f"{selected_drug} → {selected_disease}", 'font': {'size': 16}},
    gauge = {
        'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkgray"},
        'bar': {'color': color},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 0.5], 'color': '#f8f9fa'},
            {'range': [0.5, 0.75], 'color': '#fef5e7'},
            {'range': [0.75, 1], 'color': '#eafaf1'}],
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': probability}
    }
))

fig_gauge.update_layout(
    height=300,
    margin=dict(t=50, b=10, l=20, r=20),
    font=dict(family="sans-serif", color=PRIMARY_COLOR)
)

st.plotly_chart(fig_gauge, use_container_width=True)

# Confidence interpretation
st.markdown(f"""
<div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
    <h4 style="color: {color}; margin: 0 0 10px 0;">{interpretation} ({confidence} Confidence)</h4>
    <p style="color: #555; margin: 0;">
        This prediction is based on the structural relationships in the biological knowledge graph 
        and the learned patterns from known therapeutic associations.
    </p>
</div>
""", unsafe_allow_html=True)

# Explainability with SHAP
st.markdown("<h4 style='color: #2c3e50; margin-top: 1.5rem;'>Prediction Explanation</h4>", unsafe_allow_html=True)
st.markdown("""
<p style="color:#555;">
The SHAP (SHapley Additive exPlanations) values below show how each embedding dimension 
contributes to the prediction. Positive values push the prediction toward a therapeutic 
relationship, while negative values push against it.
</p>
""", unsafe_allow_html=True)

with st.spinner("Generating explanation..."):
    shap_fig = generate_shap极planation(model, input_vector.reshape(1, -1))
    st.pyplot(shap_fig)

# -- Step 4: Embedding Space Visualization --
st.markdown("<h2 class='header'>Embedding Space Analysis</h2>", unsafe_allow_html=True)
st.markdown(f"""
<p style="color:#555;">
Principal Component Analysis (PCA) projection of {actual_dim}-dimensional node embeddings 
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
for idx, node in enumerate(all_nodes):
    node_type = G.nodes[node]['type']
    ax2.scatter(
        pca[idx, 0], pca[idx, 1], 
        color=NODE_COLORS[node_type], 
        s=120,
        alpha=0.8,
        edgecolor='#2c3e50',
        linewidth=0.5
    )

# Add labels with offsets to avoid overlapping
label_positions = {}
for idx, node in enumerate(all_nodes):
    x, y = pca[idx, 0], pca[idx, 1]
    
    # Adjust position if overlapping
    offset_x, offset_y = 0.05, 0.05
    while (x + offset_x, y + offset_y) in label_positions.values():
        offset_x += 0.02
        offset_y += 0.02
    
    label_positions[node] = (x + offset_x, y + offset_y)
    ax2.annotate(node, (x, y), 
                xytext=(offset_x*40, offset_y*40),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', 
                          fc='white', 
                          alpha=0.7,
                          ec='none'))

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

# Enhanced Methodology Overview with step cards
st.markdown("""
<div class="methodology-card">
    <h3 style="color: #2c3e50; margin-top: 0;">Computational Framework</h3>
    
    <div class="step-card">
        <h4 style="margin: 0 0 8px 0; color: #2c3e50;">1. Knowledge Graph Construction</h4>
        <p style="color: #555; margin: 0;">
        Integration of pharmaceutical compounds, protein targets, and disease states 
        with established biological relationships into a structured network.
        </p>
    </div>
    
    <div class="step-card">
        <h4 style="margin: 0 0 8px 0; color: #2c3e50;">2. Node Embedding Generation</h4>
        <p style="color: #555; margin: 0;">
        Application of random walk algorithms to learn continuous vector representations 
        of entities that capture structural relationships within the knowledge graph.
        </p>
    </div>
    
    <div class="step-card">
        <h4 style="margin: 0 0 8px 0; color: #2c3e50;">3. Predictive Model Development</h4>
        <p style="color: #555; margin: 0;">
        Training of a logistic regression classifier using concatenated drug-disease 
        embedding vectors to predict therapeutic relationships.
        </p>
    </div>
    
    <div class="step-card">
        <h4 style="margin: 0 0 8px 0; color: #2c3e50;">4. Therapeutic Relationship Prediction</h4>
        <p style="color: #555; margin: 0;">
        Probabilistic assessment of novel drug-disease relationships through 
        model inference on embedding vector pairs.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Analytical Validation
st.markdown("""
<div style="background-color: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-top: 1.5rem;">
    <h3 style="color: #2c3e50; margin-top: 0;">Analytical Validation</h3>
    
    <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
        <div style="flex: 1; padding: 1rem; background-color: #eaf7f0; border-radius: 6px; margin-right: 0.5rem;">
            <h4 style="margin: 0 0 5px 0; color: #27ae60;">CV Accuracy</h4>
            <p style="font-size: 1.5rem; font-weight: bold; margin: 0; color: #2c3e50;">{cv_accuracy:.1%}</p>
        </div>
        
        <div style="flex: 1; padding: 1rem; background-color: #f0f7fa; border-radius: 6px; margin-right: 0.5rem;">
            <h4 style="margin: 0 0 5px 0; color: #3498db;">Test Accuracy</h4>
            <p style="font-size: 1.5rem; font-weight: bold; margin: 0; color: #2c3e50;">{test_accuracy:.1%}</p>
        </div>
        
        <div style="flex: 1; padding: 1rem; background-color: #f5f0f7; border-radius: 6px;">
            <h4 style="margin: 0 0 5px 0; color: #8e44ad;">Class Balance</h4>
            <p style="font-size: 1.5rem; font-weight: bold; margin: 0; color: #2c3e50;">{len(positives)}:{len(negatives)}</p>
        </div>
    </div>
    
    <p style="color: #555; margin-top: 1.5rem;">
    The predictive model was developed using {len(positives)} known therapeutic relationships 
    and {len(negatives)} negative controls, achieving a cross-validated accuracy of {cv_accuracy:.1%} 
    and test set accuracy of {test_accuracy:.1%}.
    </p>
</div>
""", unsafe_allow_html=True)

# Applications
st.markdown("""
<div style="background-color: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-top: 1.5rem;">
    <h3 style="color: #2c3e50; margin-top: 0;">Applications</h3>
    
    <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
        <div style="flex: 1; padding: 1rem; margin-right: 0.5rem;">
            <h4 style="margin: 0 0 8px 0; color: #2c3e50;">Drug Repurposing</h4>
            <p style="color: #555; margin: 0;">
            Identification of novel therapeutic indications for existing pharmaceutical compounds.
            </p>
        </div>
        
        <div style="flex: 1; padding: 1rem; margin-right: 0.5rem;">
            <h4 style="margin: 0 0 8px 0; color: #2c3e50;">Mechanism Discovery</h4>
            <p style="color: #555; margin: 0;">
            Elucidation of potential mechanisms of action through protein interaction networks.
            </p>
        </div>
        
        <div style="flex: 1; padding: 1rem;">
            <h4 style="margin: 0 0 8px 0; color: #2c3e50;">Candidate Prioritization</h4>
            <p style="color: #555; margin: 0;">
            Systematic prioritization of compounds for further experimental validation.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

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