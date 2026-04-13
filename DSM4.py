import streamlit as st
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Peer Recommender")

st.title(" Smart Peer Recommendation System")
st.markdown("Based on Discrete Structures & Graph Theory (DSGT) using the **Jaccard Index**.")
st.divider()

# 1. Expanded Universal Set and Profiles
available_skills = [
    "Java", "Python", "C++", "JavaScript", "React", "Node.js", # Programming
    "AI", "Machine Learning", "Data Science", "SQL",           # Data/AI
    "WebDev", "UI/UX Design", "Figma",                         # Web/Design
    "Cricket", "Chess", "Photography", "Music"                 # Hobbies
]

if "students" not in st.session_state:
    st.session_state.students = {
        "S1": ["Java", "Python", "Machine Learning", "AI"],
        "S2": ["JavaScript", "React", "Node.js", "WebDev"],
        "S3": ["Python", "Data Science", "SQL"],
        "S4": ["UI/UX Design", "Figma", "Photography"],
        "S5": ["Java", "C++", "Chess"],
        "S6": ["Python", "AI", "Photography", "Music"],
        "S7": ["React", "UI/UX Design", "WebDev"] # Added S7 for more complex connections
    }

# --- Sidebar for User Input ---
st.sidebar.header("Edit Student Profiles")
st.sidebar.markdown("Add or remove skills to see the graphs update in real-time.")
for student, skills in st.session_state.students.items():
    st.session_state.students[student] = st.sidebar.multiselect(
        f"{student}'s Skills", 
        options=available_skills, 
        default=skills
    )

# 2. Jaccard Similarity Function
def jaccard_index(list_a, list_b):
    set_a, set_b = set(list_a), set(list_b)
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if len(union) > 0 else 0.0

student_names = list(st.session_state.students.keys())
n = len(student_names)

# 3. Build Adjacency Matrix
adjacency_matrix = np.zeros((n, n))
for i, j in itertools.combinations(range(n), 2):
    score = jaccard_index(st.session_state.students[student_names[i]], st.session_state.students[student_names[j]])
    adjacency_matrix[i][j] = score
    adjacency_matrix[j][i] = score

# 4. Build Graph
G = nx.Graph()
G.add_nodes_from(student_names)
for i, j in itertools.combinations(range(n), 2):
    weight = adjacency_matrix[i][j]
    if weight > 0:
        G.add_edge(student_names[i], student_names[j], weight=weight)

# --- Main Page Visualizations ---
col1, col2 = st.columns(2)

with col1:
    st.subheader(" Adjacency Matrix (Heatmap)")
    fig2, ax2 = plt.subplots(figsize=(6, 6))

    fig2.patch.set_alpha(0.0) 
    ax2.patch.set_alpha(0.0)
    
    display_matrix = adjacency_matrix.copy()
    np.fill_diagonal(display_matrix, np.nan)
    
    # Using a sleeker, modern colormap ("Blues")
    im = ax2.imshow(display_matrix, cmap="Blues", vmin=0, vmax=1)
    ax2.set_xticks(range(n)); ax2.set_yticks(range(n))
    ax2.set_xticklabels(student_names); ax2.set_yticklabels(student_names)
    

    for i in range(n):
        for j in range(n):
            if i != j:
                val = adjacency_matrix[i][j]
                text_color = "white" if val > 0.6 else "black"
                ax2.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color)
                
    plt.colorbar(im, ax=ax2, shrink=0.8)
    st.pyplot(fig2)

with col2:
    st.subheader(" Peer Recommendation Graph")
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    
    # Make background transparent
    fig1.patch.set_alpha(0.0)
    ax1.patch.set_alpha(0.0)
    
    pos = nx.spring_layout(G, weight="weight", seed=42)
    edges = list(G.edges(data=True))
    weights = [d["weight"] for _, _, d in edges]
    
    norm = Normalize(vmin=0, vmax=1)
    colormap = plt.colormaps["Blues"] # Match heatmap theme
    edge_colors = [colormap(norm(w)) for w in weights]
    max_w = max(weights) if weights else 1
    edge_widths = [1 + (w / max_w) * 6 for w in weights]
    
    # Draw graph with modern styling
    nx.draw_networkx(
        G, pos, ax=ax1, with_labels=True, 
        node_color="#00C4B4", # A vibrant, modern teal color
        edge_color=edge_colors, width=edge_widths, 
        node_size=1000, font_color="white", font_weight="bold"
    )
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax1, font_size=8)
    
    ax1.axis("off")
    st.pyplot(fig1)