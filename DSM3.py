import streamlit as st
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

st.set_page_config(layout="wide")
st.title("🤝 Smart Peer Recommendation System")
st.markdown("Based on Discrete Structures & Graph Theory (DSGT) using the **Jaccard Index**.")

# 1. Define the Universal Set and default student profiles in Session State
available_skills = ["Java", "Python", "C++", "AI", "WebDev", "Data Science", "Cricket", "Chess", "Design"]

if "students" not in st.session_state:
    st.session_state.students = {
        "S1": ["Java", "Python", "C++", "AI"],
        "S2": ["Java", "Python", "C++", "WebDev"],
        "S3": ["Python", "Data Science"],
        "S4": ["Cricket", "Chess"],
        "S5": ["Java", "Chess"],
        "S6": ["Java", "Python"]
    }

# --- Sidebar for User Input ---
st.sidebar.header("Edit Student Profiles")
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
    st.subheader("Adjacency Matrix (Heatmap)")
    fig2, ax2 = plt.subplots()
    display_matrix = adjacency_matrix.copy()
    np.fill_diagonal(display_matrix, np.nan)
    
    im = ax2.imshow(display_matrix, cmap="YlOrRd", vmin=0, vmax=1)
    ax2.set_xticks(range(n)); ax2.set_yticks(range(n))
    ax2.set_xticklabels(student_names); ax2.set_yticklabels(student_names)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                ax2.text(j, i, f"{adjacency_matrix[i][j]:.2f}", ha="center", va="center")
                
    plt.colorbar(im, ax=ax2)
    st.pyplot(fig2)

with col2:
    st.subheader("Peer Recommendation Graph")
    fig1, ax1 = plt.subplots()
    pos = nx.spring_layout(G, weight="weight", seed=42)
    edges = list(G.edges(data=True))
    weights = [d["weight"] for _, _, d in edges]
    
    norm = Normalize(vmin=0, vmax=1)
    colormap = plt.colormaps["YlOrRd"]
    edge_colors = [colormap(norm(w)) for w in weights]
    max_w = max(weights) if weights else 1
    edge_widths = [1 + (w / max_w) * 5 for w in weights]
    
    nx.draw_networkx(
        G, pos, ax=ax1, with_labels=True, node_color="skyblue",
        edge_color=edge_colors, width=edge_widths, node_size=800
    )
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax1)
    ax1.axis("off")
    st.pyplot(fig1)