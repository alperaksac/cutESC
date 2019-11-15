"""
CutESC: Cutting Edge Spatial Clustering Technique based on Proximity Graphs
"""

from __future__ import print_function

import itertools
import math
import multiprocessing as mp
import time

import networkx as nx
import numpy as np
import numpy.matlib as npm
from kneed import KneeLocator
from scipy.spatial import Delaunay

# Author: Alper Aksac <alperaksac@gmail.com>
#
# License: MIT


def cutESC(X, alpha=1.0, beta=1.0, allow_outliers=False,
           verbose=False):
    """Perform CutESC clustering

    Parameters
    ----------
    X : ndarray of floats, shape (npoints, ndim)
        Coordinates of data points to triangulate

    alpha : float, optional (default=1.0)
        Tuning parameter for the local searching. 
        Decreasing the value will allow to identify 
        more inconsistent edges and remove them. Thus, 
        It might generate too many clusters.

    beta : float, optional (default=1.0)
        Tuning parameter for the local inner searching. 
        Decreasing the value will allow to identify 
        more inconsistent edges and remove them. Thus, 
        It might generate too many clusters.

    allow_outliers : bool, optinal (default=False)
        Whether to accept the outliers as clusters
        rather than labelling them -1.

    verbose : bool, optinal (default=False)
        Whether to print messages to console.

    Returns
    -------
    labels : ndarray, shape (n_samples, )
        Cluster labels for each point. Outliers are 
        given the label -1.

    comps : generator of lists
        A list of nodes for each component of G.

    """

    if type(X) is not np.ndarray:
        raise ValueError('Coordinates of data points must be '
                         'ndarray of floats, shape (npoints, ndim)!')

    process_times, e_counts = [], []

    start = time.time()
    G = build_graph(X)
    end = time.time()
    process_times.append(end-start)
    e_counts.append(G.number_of_edges())

    start = time.time()
    compute_gabriel(G)
    end = time.time()
    process_times.append(end-start)
    e_counts.append(G.number_of_edges())

    start = time.time()
    global_edges(G)
    end = time.time()
    process_times.append(end-start)
    e_counts.append(G.number_of_edges())

    start = time.time()
    local_edges(G, alpha)
    end = time.time()
    process_times.append(end-start)
    e_counts.append(G.number_of_edges())

    start = time.time()
    local_inner_edges(G, beta)
    end = time.time()
    process_times.append(end-start)
    e_counts.append(G.number_of_edges())

    start = time.time()
    labels_pred, comps = find_components(G, allow_outliers)
    end = time.time()
    process_times.append(end-start)

    if verbose:
        print(60 * '*')
        print('Forming Delaunay Graph: %.3f (sec)' % process_times[0])
        print('Forming Gabriel Graph: %.3f (sec)' % process_times[0])
        print('Identifying Global Long Edges: %.3f (sec)' % process_times[0])
        print('Identifying Local Long Edges: %.3f (sec)' % process_times[0])
        print('Identifying Local Inner Long Edges: %.3f (sec)' %
              process_times[0])
        print('Finding Components: %.3f (sec)' % process_times[0])
        print('Total Process: %.3f (sec)' % sum(process_times))
        print(60 * '*')
        print(60 * '*')
        print('Step 1, # number of edges: %d' % e_counts[0])
        print('Step 2, # number of edges: %d' % e_counts[1])
        print('Step 3, # number of edges: %d' % e_counts[2])
        print('Step 4, # number of edges: %d' % e_counts[3])
        print('Step 5, # number of edges: %d' % e_counts[4])
        print(60 * '*')
    return labels_pred, comps


def build_graph(points):
    """
    build the graph from delaunay triangles
    """

    # create the delaunay triangulation, "QJ" option allows
    # to create delaunay triangles from duplicate points
    tri = Delaunay(points, qhull_options='QJ Pp')
    edges = []

    # create edges
    for v in tri.simplices:
        for indices in itertools.combinations(v, 2):
            edges.append(indices)
    graph = nx.Graph(edges)

    nodes = dict(graph.nodes())
    # create nodes
    for i in nodes:
        nodes[i]['pos'] = np.array(points[i])

    adjacency = dict(graph.adjacency())

    removed_edges = []
    # calculate edge weights
    for i in nodes:
        for j in adjacency[i]:
            weight = np.linalg.norm(nodes[i]['pos'] - nodes[j]['pos'])
            if weight != 0:
                graph.edges[(i, j)]['weight'] = weight
            else:
                removed_edges.append((i, j))
    graph.remove_edges_from(removed_edges)
    return graph


def check_edge(i):
    """
    check the edge's geometric condition for the gabriel graph
    """

    removed_edges = []
    for j in adjacency[i]:
        # calculate Euclidean distance between nodes i and j
        dist_ij = np.linalg.norm(nodes[i]['pos'] - nodes[j]['pos'])
        for k in adjacency[i]:
            if j == k:
                continue
            dist_ik = np.linalg.norm(nodes[i]['pos'] - nodes[k]['pos'])
            dist_jk = np.linalg.norm(nodes[j]['pos'] - nodes[k]['pos'])

            # gabriel graph condition
            if dist_ij**2 > dist_ik**2 + dist_jk**2:
                removed_edges.append((i, j))
                break
    return removed_edges


def compute_gabriel(graph):
    """
    compute the gabriel graph
    """

    global nodes, adjacency
    nodes, adjacency = dict(graph.nodes()), dict(graph.adjacency())

    pool = mp.Pool(processes=mp.cpu_count()-1)
    removed_edges = pool.map(check_edge, nodes)
    removed_edges = set(itertools.chain.from_iterable(removed_edges))
    pool.close()
    # removing edges from the graph
    graph.remove_edges_from(removed_edges)


def global_edges(graph):
    """
    identifying globally long edges
    """

    v_count = graph.number_of_nodes()
    edges, nodes, adjacency = dict(graph.edges()), dict(
        graph.nodes()), dict(graph.adjacency())
    edge_weights = list(nx.get_edge_attributes(graph, 'weight').values())
    local_means = [0] * v_count

    for i in nodes:
        local_mean = 0
        for j in adjacency[i]:
            local_mean += graph.edges[i, j]['weight']
        if len(adjacency[i]) != 0:
            local_mean /= len(adjacency[i])
        local_means[i] = local_mean

    global_mean = np.mean(edge_weights)
    global_std = sum([math.pow(global_mean - local_means[i], 2)
                      for i in nodes])
    global_std = math.sqrt(global_std / (v_count-1))

    cut = []
    for i in range(v_count):
        if local_means[i] != 0:
            cut.append(global_mean + (global_mean *
                                      global_std / local_means[i]))
        else:
            cut.append(0)

    # remove edges
    removed_edges = [e for e in edges if edges[e]['weight']
                     >= cut[e[0]] or edges[e]['weight'] >= cut[e[1]]]
    graph.remove_edges_from(removed_edges)


def local_edges(graph, alpha=1.0):
    """
    identifying locally long edges
    """

    v_count = graph.number_of_nodes()
    labels, comps = find_components(graph)
    edges, nodes, adjacency = dict(graph.edges()), dict(
        graph.nodes()), dict(graph.adjacency())
    local_stds = [0] * v_count
    comps_means = [0] * len(comps)

    for i in nodes:
        neigh_count, local_mean = 0, 0
        # first degree neighbor
        for j in adjacency[i]:
            local_mean += graph.edges[i, j]['weight']
            neigh_count += 1
            # second degree neighbor
            for k in adjacency[j]:
                local_mean += graph.edges[j, k]['weight']
                neigh_count += 1
        if neigh_count != 0:
            local_mean /= neigh_count
        comps_means[labels[i]] += local_mean

    comps_means = [val / len(comps[i]) for i, val in enumerate(comps_means)]

    for i in nodes:
        neigh_count, local_std = 0, 0
        # first degree neighbor
        for j in adjacency[i]:
            weight = graph.edges[i, j]['weight']
            local_std += math.pow(comps_means[labels[i]] - weight, 2)
            neigh_count += 1
            # second degree neighbor
            for k in adjacency[j]:
                weight = graph.edges[j, k]['weight']
                local_std += math.pow(comps_means[labels[i]] - weight, 2)
                neigh_count += 1

        if neigh_count != 0:
            local_std = math.sqrt(local_std / (neigh_count-1))
        local_stds[i] = local_std

    removed_edges = []
    for e in edges:
        try:
            cut1 = comps_means[labels[e[0]]] + alpha * local_stds[e[0]] * math.exp(
                comps_means[labels[e[0]]] / graph.edges[e]['weight'])
        except OverflowError:
            cut1 = float('inf')
        try:
            cut2 = comps_means[labels[e[1]]] + alpha * local_stds[e[1]] * math.exp(
                comps_means[labels[e[1]]] / graph.edges[e]['weight'])
        except OverflowError:
            cut2 = float('inf')
        if edges[e]['weight'] >= cut1 or edges[e]['weight'] >= cut2:
            removed_edges.append(e)
    graph.remove_edges_from(removed_edges)


def local_inner_edges(graph, beta=1.0):
    """
    identifying local inner long edges
    """

    v_count = graph.number_of_nodes()
    edges, nodes, adjacency = dict(graph.edges()), dict(
        graph.nodes()), dict(graph.adjacency())
    local_means, local_stds = [0] * v_count, [0] * v_count

    for i in nodes:
        neigh_count, local_mean = 0, 0
        # first degree neighbor
        for j in adjacency[i]:
            local_mean += graph.edges[i, j]['weight']
            neigh_count += 1
            # second degree neighbor
            for k in adjacency[j]:
                local_mean += graph.edges[j, k]['weight']
                neigh_count += 1
        if neigh_count != 0:
            local_mean /= neigh_count
        local_means[i] = local_mean

    for i in nodes:
        neigh_count, local_std = 0, 0
        # first degree neighbor
        for j in adjacency[i]:
            weight = graph.edges[i, j]['weight']
            local_std += math.pow(local_means[i] - weight, 2)
            neigh_count += 1
            # second degree neighbor
            for k in adjacency[j]:
                weight = graph.edges[j, k]['weight']
                local_std += math.pow(local_means[i] - weight, 2)
                neigh_count += 1

        if neigh_count != 0:
            local_std = math.sqrt(local_std / (neigh_count-1))
        local_stds[i] = local_std

    removed_edges = []
    for e in edges:
        try:
            cut1 = local_means[e[0]] + beta * local_stds[e[0]] * math.exp(
                local_means[e[0]] / graph.edges[e]['weight'])
        except OverflowError:
            cut1 = float('inf')
        try:
            cut2 = local_means[e[1]] + beta * local_stds[e[1]] * math.exp(
                local_means[e[1]] / graph.edges[e]['weight'])
        except OverflowError:
            cut2 = float('inf')
        if edges[e]['weight'] >= cut1 or edges[e]['weight'] >= cut2:
            removed_edges.append(e)
    graph.remove_edges_from(removed_edges)


def find_components(graph, allow_outliers=False):
    """
    find the strongly connected components
    """
    comps = [c for c in sorted(
        nx.connected_components(graph), key=len, reverse=True)]
    labels = [0] * graph.number_of_nodes()
    for i, c in enumerate(comps):
        for n in c:
            labels[n] = i

    if allow_outliers:
        hist = [len(v) for v in comps]
        x_axis = list(range(len(hist)))
        kn = KneeLocator(x_axis, hist, S=1.0,
                         curve='convex', direction='decreasing')
        idx = kn.knee
        for i, c in enumerate(comps[idx:]):
            for n in c:
                labels[n] = -1
    return np.array(labels), comps
