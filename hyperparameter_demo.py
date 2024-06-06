import umap
import umap.plot
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt

swissroll, labels = datasets.make_swiss_roll(
    n_samples=2000, noise=0.1, random_state=69, hole=True
)

sns.set_theme(context="paper", style="white")

numbers_min_dist = [0.01,0.1,0.5,1]
reducers_min_dist = map(lambda min_dist: umap.UMAP(min_dist=min_dist, random_state=1), numbers_min_dist)

numbers_n_neighbors = [4,15,80,500]
reducers_n_neighbors = map(lambda n_neighbors: umap.UMAP(n_neighbors=n_neighbors, random_state=1), numbers_n_neighbors)

numbers_n_epochs = [1,10,200,2000]
reducers_n_epochs = map(lambda n_epochs: umap.UMAP(n_epochs=n_epochs, random_state=1), numbers_n_epochs)

n_rows = 3
n_cols = 4
ax_index = 1
ax_list = []

plt.figure(figsize=(10, 8))
plt.subplots_adjust(
    left=0.02, right=0.98, bottom=0.001, top=0.96, wspace=0.05, hspace=0.01
)

def draw_comparison(reducers, name, values):
    global ax_index   

    for reducer in reducers:
        embedding = reducer.fit_transform(swissroll)
        ax = plt.subplot(n_rows, n_cols, ax_index)
        ax.scatter(*embedding.T, s=10, c=labels, alpha=0.5)
        ax_list.append(ax)
        ax.set_xlabel(f"{name}={values[(ax_index - 1) % n_cols]}", size=16)
        ax.xaxis.set_label_position("top")

        ax_index += 1

draw_comparison(reducers_min_dist, "min-dist", numbers_min_dist)
draw_comparison(reducers_n_neighbors, "n_neighbors", numbers_n_neighbors)
draw_comparison(reducers_n_epochs, "n_epochs", numbers_n_epochs)

plt.setp(ax_list, xticks=[], yticks=[])
plt.tight_layout()
plt.savefig(f"swissroll_cmp")

