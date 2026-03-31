import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# ─────────────────────────────────────────────────────────────
# SECTION 1: Define the Universal Set and Student Profiles
# ─────────────────────────────────────────────────────────────

U = {"Java", "Python", "C++", "AI", "WebDev", "Data Science", "Cricket", "Chess"}

students = {
    "S1": {"Java", "Python", "C++", "AI"},
    "S2": {"Java", "Python", "C++", "WebDev"},
    "S3": {"Python", "Data Science"},
    "S4": {"Cricket", "Chess"},
    "S5": {"Java", "Chess"},
    "S6": {"Java", "Python"},
}

student_names = list(students.keys())
n = len(student_names)

# ─────────────────────────────────────────────────────────────
# SECTION 2: Jaccard Similarity
# ─────────────────────────────────────────────────────────────

def jaccard_index(set_a, set_b):
    intersection = set_a & set_b
    union = set_a | set_b

    if len(union) == 0:
        return 0.0

    return len(intersection) / len(union)

# ─────────────────────────────────────────────────────────────
# SECTION 3: Adjacency Matrix
# ─────────────────────────────────────────────────────────────

adjacency_matrix = np.zeros((n, n))

print("=" * 60)
print("  JACCARD SIMILARITY SCORES — All Student Pairs")
print("=" * 60)

for i, j in itertools.combinations(range(n), 2):
    si_name = student_names[i]
    sj_name = student_names[j]

    si = students[si_name]
    sj = students[sj_name]

    score = jaccard_index(si, sj)

    adjacency_matrix[i][j] = score
    adjacency_matrix[j][i] = score

    flag = ""
    if score == 0:
        flag = "  ← NO EDGE"
    elif score >= 0.5:
        flag = "  ← HIGH MATCH ★"

    print(f"J({si_name}, {sj_name}) = {score:.4f}{flag}")

print("\nAdjacency Matrix:\n", adjacency_matrix)

# ─────────────────────────────────────────────────────────────
# SECTION 4: Graph Creation
# ─────────────────────────────────────────────────────────────

G = nx.Graph()

for name in student_names:
    G.add_node(name)

for i, j in itertools.combinations(range(n), 2):
    weight = adjacency_matrix[i][j]
    if weight > 0:
        G.add_edge(student_names[i], student_names[j], weight=weight)

print(f"\nGraph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# ─────────────────────────────────────────────────────────────
# SECTION 5: Visualization
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ---- GRAPH ----
ax1 = axes[0]

pos = nx.spring_layout(G, weight="weight", seed=42)

edges = G.edges(data=True)
weights = [d["weight"] for _, _, d in edges]

# UPDATED (no warning)
colormap = plt.get_cmap("plasma")

norm = Normalize(vmin=0, vmax=max(weights))
edge_colors = [colormap(norm(w)) for w in weights]
edge_widths = [1 + w * 5 for w in weights]

nx.draw(G, pos,
        ax=ax1,
        with_labels=True,
        node_color="skyblue",
        edge_color=edge_colors,
        width=edge_widths,
        node_size=800,
        font_weight="bold")

ax1.set_title("Peer Recommendation Graph")

# ---- HEATMAP ----
ax2 = axes[1]

display_matrix = adjacency_matrix.copy()
np.fill_diagonal(display_matrix, np.nan)

im = ax2.imshow(display_matrix, cmap="YlOrRd", vmin=0, vmax=1)

ax2.set_xticks(range(n))
ax2.set_yticks(range(n))
ax2.set_xticklabels(student_names)
ax2.set_yticklabels(student_names)

for i in range(n):
    for j in range(n):
        if i != j:
            ax2.text(j, i, f"{adjacency_matrix[i][j]:.2f}",
                     ha="center", va="center")

plt.colorbar(im, ax=ax2)
ax2.set_title("Adjacency Matrix")

# ─────────────────────────────────────────────────────────────
# SAVE + SHOW
# ─────────────────────────────────────────────────────────────

plt.tight_layout()

# ✅ FIXED PATH (works on Windows)
plt.savefig("peer_recommendation_graph.png", dpi=150)

plt.show()

print("\nGraph saved as: peer_recommendation_graph.png")