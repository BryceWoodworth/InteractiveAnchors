__author__ = 'ephax'

import numpy
import time
from scipy.optimize import linprog
from sklearn import random_projection
from sklearn.cluster import KMeans


def profile(V, k):
    convex = numpy.random.uniform(size=(V, V))

    # random projection
    if k < V:
        GRP = random_projection.GaussianRandomProjection(n_components=k)
        convex = GRP.fit_transform(convex).transpose()

    sums = numpy.sum(convex, axis=0, keepdims=True)
    convex /= sums
    convex = numpy.vstack((convex, numpy.ones(V)))

#    c = numpy.zeros(V-1)
    c = numpy.zeros(V-1)
    extremal_count = 0
    extremal_pts = None
    indices = []
    start_time = time.time()
    for i in range(V):
        # check if the i'th column lies in the convex hull of others
        A = numpy.hstack((convex[:, :i], convex[:, i+1:]))
        target = convex[:, i]
        if not linprog(c, A_eq=A, b_eq=target).success:
            extremal_count += 1
            indices.append(i)
            if extremal_pts is None:
                extremal_pts = [convex[:, i]]
            else:
                extremal_pts = numpy.concatenate((extremal_pts, [convex[:, i]]), axis=0)
    return time.time() - start_time, extremal_count, zip(indices, extremal_pts)

profile(100, 10)

# def compute_hull(V, k):
#     convex = numpy.random.uniform(size=(V, V))
#
#     # random projection
#     if k < V:
#         GRP = random_projection.GaussianRandomProjection(n_components=k)
#         convex = GRP.fit_transform(convex).transpose()
#
#     sums = numpy.sum(convex, axis=0, keepdims=True)
#     convex /= sums
#     convex = numpy.vstack((convex, numpy.ones(V)))


# only run this if we have more candidate points than topics!
# if not, your projection dimension is probably too low
def compute_means(extremal, topics):
    start_time = time.time()
    clusterer = KMeans(n_clusters=topics)
    indices, points = zip(*extremal)
    labels = clusterer.fit_predict(points)
    centroids = clusterer.cluster_centers_
    # for each cluster, compute the starting anchor word by picking
    # the word with the largest minimal distance to another cluster centroid
    anchors = numpy.array([(-1, float("inf")) for _ in range(topics)])
    for (index, point, label) in zip(indices, points, labels):
        min_dist = float("inf")
        for i in range(topics):
            if i != label:  # don't consider the centroid we belong to
                dist = numpy.linalg.norm(point - centroids[i])
                if dist < min_dist:
                    min_dist = dist

        if min_dist < anchors[label][1]:
            anchors[label] = (index, min_dist)

    return anchors, time.time() - start_time


def performance_chart(space, num_topics, start, end, step, num_trials):
    for i in range(start, end, step):
        total_extremal_time = 0.0
        total_anchor_time = 0.0
        total_extremal = 0
        anchor_trials = 0
        for j in range(num_trials):
            (trial_extremal_time, trial_extremal, extremal_pts) = profile(space, i)
            total_extremal_time += trial_extremal_time
            total_extremal += trial_extremal
            if trial_extremal < num_topics:
                print("(%d, %d) -> too few extremal points (projection dim too low)" %
                      space, i)
            else:
                anchors, trial_anchor_time = compute_means(extremal_pts, num_topics)
                total_anchor_time += trial_anchor_time
                anchor_trials += 1
        print("(%d, %d) -> %.2f, average %.1f extremal\n\t%.2f KMeans+anchor time" %
              (space, i, total_extremal_time / num_trials, total_extremal / num_trials,
               total_anchor_time / anchor_trials))
