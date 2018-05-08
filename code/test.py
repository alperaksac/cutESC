import numpy as np

import benchmarks
import cutESC as cut

data = np.loadtxt(fname='data/compound.txt', delimiter=',')
X, labels_true = data[:, [0, 1]], data[:, 2]

labels_pred, comps = cut.cutESC(X, allow_outliers=True, verbose=True)
results = benchmarks.benchmarks(X, labels_true, labels_pred, verbose=True)
plt = benchmarks.draw_clusters(X, labels_pred)
benchmarks.write_plot('results/compound.png', plt)
plt.show()

# or you can the functions one by one
data = np.loadtxt(fname='data/t8.8k.txt', delimiter=',')
X, labels_true = data[:, [0, 1]], data[:, 2]

G = cut.build_graph(X)
cut.compute_gabriel(G)
cut.global_edges(G)
cut.local_edges(G)
cut.local_inner_edges(G)
labels_pred, comps = cut.find_components(G, allow_outliers=True)

results = benchmarks.benchmarks(X, labels_true, labels_pred, verbose=True)
plt = benchmarks.draw_clusters(X, labels_pred)
benchmarks.write_plot('results/t8.8k.png', plt)
plt.show()