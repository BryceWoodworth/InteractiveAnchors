import itertools

__author__ = 'Bryce Woodworth'

import numpy
import numpy.random
import time
from math import log
from numpy.core.umath_tests import inner1d
from scipy.optimize import linprog
from sklearn import random_projection
from sklearn.cluster import KMeans

def _farthest_span(span, options, chosen):
    farthest = -1
    index = -1
    for i in range(len(options)):
        candidate = options[i]
        for vec in span:
            # subtract off the projection of candidate onto vec
            projection = numpy.dot(candidate, vec) * vec
            candidate = candidate - projection
        dist = numpy.linalg.norm(candidate)
        if dist > farthest:
            farthest = dist
            index = i
            orthonormal = candidate / dist
    return index, orthonormal


def anchor_selection(Q, k):
    """Finds anchor words from the given word-word correlation matrix.
    This algorithm assumes that there are at least k anchor words and
    that the rest of the rows in Q lie (almost) in their convex hull.
    Input:
    Q: the row-normalized word co-occurrence matrix,
    k: the number of anchor words to find

    Output:
    A numpy array of the rows of Q corresponding to anchor words

    NOTE: Right now this does not take advantage of the FJLT
    algorithm for random projection.
    """
    # Initialize anchor list with the index of the largest norm in reduced matrix
    first_anchor_index = numpy.argmax(numpy.linalg.norm(Q, ord=2, axis=1))
    anchor_indices = [first_anchor_index]
    anchors = [Q[first_anchor_index] / numpy.linalg.norm(Q[first_anchor_index])]

    while len(anchors) < k:
        # Find the row farthest from the span of other anchors
        selection, anchor = _farthest_span(anchors, Q, anchor_indices)
        anchor_indices.append(selection)
        anchors.append(anchor)

    # Correction phase
    for i in range(len(anchors)):
        anchor_indices.pop(0)
        anchors.pop(0)
        selection, anchor = _farthest_span(anchors, Q, anchor_indices)
        anchor_indices.append(selection)
        anchors.append(anchor)

    anchors = numpy.asarray([Q[i] for i in anchor_indices])
    return anchors, anchor_indices


# TODO: is projecting onto the number of topics the right choice?
def extremal_anchors(Q, k, proj):
    """Finds anchor words form the given word-word correlation matrix.
    Finds anchor words by first doing a random projection and ruling out
    any points that lie in the convex hull of other points (as they cannot
    themselves be part of the convex hull). Then we run K-Means to cluster
    into k clusters and pick one anchor word per cluster. The primary benefit
    of using this function is that it gives access to cluster information
    that can be used in an interactive setting to manipulate the selected anchor
    words.
    Input:
    Q: the row-normalized word co-occurrence matrix,
    k: the number of anchor words to find,
    proj: the number of dimensions to use in the random projection

    Output:
    A numpy array of the rows of Q corresponding to anchor words,
    and their indices.
    """
    # random projection
    GRP = random_projection.GaussianRandomProjection(n_components=proj)
    Q_red = GRP.fit_transform(Q).transpose()

    # renormalize
    sums = numpy.sum(Q_red, axis=0, keepdims=True)
    Q_red /= sums

    # add a row of 1's at the bottom to keep the mixing weights a simplex
    Q_red = numpy.vstack((Q_red, numpy.ones(Q.shape[0])))

    # use an empty optimization function since we only need a feasible point
    c = numpy.zeros(Q.shape[0]-1)

    extremal_count = 0
    extremal_pts = None
    indices = []

    for i in range(Q.shape[0]):
        # check if the i'th column lies in the convex hull of others
        A = numpy.hstack((Q_red[:, :i], Q_red[:, i+1:]))
        target = Q_red[:, i]
        if not linprog(c, A_eq=A, b_eq=target).success:
            extremal_count += 1
            indices.append(i)
            if extremal_pts is None:
                extremal_pts = [Q_red[:, i]]
            else:
                extremal_pts = numpy.concatenate((extremal_pts, [Q_red[:, i]]), axis=0)

    # Now that we have our candidate extremal points, cluster with K-Means
    #return time.time() - start_time, extremal_count, zip(indices, extremal_pts)

    # TODO: handle case where we have fewer extremal points than topics
    clusterer = KMeans(n_clusters=k)
    # TODO: give access to the clusters for use in interactivity
    labels = clusterer.fit_predict(extremal_pts)
    centroids = clusterer.cluster_centers_
    # for each cluster, compute the starting anchor word by picking
    # the word with the largest minimal distance to another cluster centroid
    # TODO: this is an important step, is this the best way?
    anchors = numpy.array([(-1, float("inf")) for _ in range(k)])
    for (index, point, label) in zip(indices, extremal_pts, labels):
        min_dist = float("inf")
        for i in range(k):
            if i != label:  # don't consider the centroid we belong to
                dist = numpy.linalg.norm(point - centroids[i])
                if dist < min_dist:
                    min_dist = dist

        if min_dist < anchors[label][1]:
            anchors[label] = (index, min_dist)

    # extract the found anchors from the Q matrix and return them
    anchor_indices, _ = zip(*anchors)
    anchor_indices = [int(x) for x in anchor_indices]
    return Q[anchor_indices], anchor_indices

# TODO: evaluate this as a means to add a sparse solution
#kernel_terms = [None, None, None]
#def l2_loss(targets, convex, weight, test_weight=None):
#    """
#    :param targets:         the matrix of values to reconstruct
#    :param convex:          the matrix of the convex set we are reconstructing with
#    :param weight:          the current estimate to optimize
#    :param test_weight:     the estimate to check in line search
#    :return:                the gradient, and if we pass in an i, an objective for that i
#
#    Preconditions:  Should never set i or weight_update on the first call
#                    if weight_update is set, i should be too
#    """
#    if kernel_terms[0] is None:
#        kernel_terms[0] = inner1d(targets, targets)
#        kernel_terms[1] = -2 * numpy.dot(targets, numpy.transpose(convex))
#        kernel_terms[2] = numpy.dot(convex, numpy.transpose(convex))
#
#    if test_weight is None:
#        grad = kernel_terms[1][i] + numpy.dot(2 * kernel_terms[2], weight)
#
#        objective = kernel_terms[0][i] + numpy.dot(weight, kernel_terms[1][i]) + \
#                    numpy.dot(weight, numpy.transpose(numpy.dot(kernel_terms[2], weight)))
#
#    else:
#        weighted_kernel_term_3 = numpy.dot(kernel_terms[2], test_weight)
#        grad = kernel_terms[3][i] + (2 * weighted_kernel_term_3)
#
#        objective = kernel_terms[0][i] + numpy.dot(test_weight, kernel_terms[1][i]) + \
#                         numpy.dot(test_weight, weighted_kernel_term_3.transpose())
#
#    return grad, objective


# TODO: add multicore support
def exponentiated_gradients(targets, convex, epsilon=1e-7, max_iters=10000, max_steps=20, starting_step=1, decay=2.0, c1=1e-4, c2=0.75, num_threads=4):
    """Finds the best reconstruction of target as a convex combination
    of other vectors.

    Inputs:
    targets:    The matrix of values to reconstruct
    convex:     The matrix of the convex set we are reconstructing with
    epsilon:    A bound on the convergence
    max_iters:  The maximal number of gradient steps made for a given row before giving up
    max_steps:  The maximal number of trials in the line search before giving up
    starting_step:  The size of the first trial step in the line search
    decay:      The ratio by which line search is modified
    c1:         The sufficient decrease parameter for line search
    c2:         The curvature parameter for line search
    num_threads:    The number of threads to use (currently unused)

    Outputs:
    The reconstruction weights
    The number of converged rows
    The total number of iterations run
    The average number of steps per iteration
    The maximal number of steps for any iteration
    """
    assert 0 < c1 < c2 < 1

    numpy.seterr(all='raise', under='ignore')
    MAX_CHANGE = 709.7  # this is the maximal number we can use on exp without overflow

    (k, V) = convex.shape

    weights = numpy.empty((V, k))

    total_error = 0

    kernel_term_1 = inner1d(targets, targets)
    kernel_term_2 = -2 * numpy.dot(targets, numpy.transpose(convex))
    kernel_term_3 = numpy.dot(convex, numpy.transpose(convex))

    converged = 0
    total_iters = 0
    total_steps = 0
    max_num_steps = -1

    for i in range(V):
        weight = numpy.ones(k) / k
        iters = 0

        line_step = starting_step

        # Run each iteration until convergence or max_iters
        while iters < max_iters:
            iters += 1
            grad = kernel_term_2[i] + numpy.dot(2 * kernel_term_3, weight)

            # Indicates whether we increased or decreased the step last iteration (avoids infinite-loops)
            prev_direction = 0

            # Get the baseline error
            base_objective = kernel_term_1[i] + numpy.dot(weight, kernel_term_2[i]) + \
                             numpy.dot(weight, numpy.transpose(numpy.dot(kernel_term_3, weight)))

            # Get the maximum safe step length
            if numpy.min(grad) < 0:
                max_step = -1 * MAX_CHANGE / numpy.min(grad)
            else:
                max_step = float('inf')

            # Perform line search to find a step length that meets the Wolfe criteria
            for steps in range(1, max_steps+1):
                total_steps += 1
                change = numpy.exp(-1 * line_step * grad)
                test_weight = numpy.multiply(weight, change)
                try:
                    test_weight /= numpy.sum(test_weight)
                except FloatingPointError:
                    # Our step was too high and we set everything to 0
                    line_step /= decay
                    prev_direction = -1
                    continue

                weighted_kernel_term_3 = numpy.dot(kernel_term_3, test_weight)
                test_objective = kernel_term_1[i] + numpy.dot(test_weight, kernel_term_2[i]) + \
                                 numpy.dot(test_weight, weighted_kernel_term_3.transpose())

                test_grad = kernel_term_2[i] + (2 * weighted_kernel_term_3)

                directional_derivative = numpy.dot(grad, test_weight - weight)
                test_directional_derivative = numpy.dot(test_grad, test_weight - weight)

                # If we fail Wolfe condition 1, sufficient decrease (too far)
                if not test_objective <= base_objective + (c1 * line_step * directional_derivative):
                    line_step /= decay
                    prev_direction = -1

                # If we fail Wolfe condition 2, curvature (not far enough)
                elif (not abs(test_directional_derivative) <= -1 * c2 * directional_derivative) and \
                        (prev_direction >= 0):
                    # If this is the best we can do, deal with it
                    if line_step >= max_step:
                        break
                    line_step *= decay
                    prev_direction = 1

                    # Don't go over the boundary
                    if line_step > max_step:
                        line_step = max_step

                # We meet both criteria!
                else:
                    break

            if steps > max_num_steps:
                max_num_steps = steps

            weight = test_weight

            mew = numpy.min(test_grad)
            lamb = test_grad - mew
            # if numpy.dot(lamb, weight) < epsilon or abs(1 - numpy.max(change)) < 1e-10:
            if numpy.dot(lamb, weight) < epsilon:
                converged += 1
                break
        weights[i] = weight
        total_error += test_objective
        total_iters += iters
    print("Average reconstruction error: %.7f" % (total_error/V))
    return weights, converged, total_iters, total_steps / total_iters, max_num_steps


def recover(Q, S, priors):
    """Takes the anchor words of Q and approximates the word-topic
    matrix A !!!and topic-topic matrix R!!!

    Inputs:
        Q: the row-normalized word-word co-occurrence counts
        S: a list of indices for anchor words
        priors: The prior probabilities of each word in Q
        epsilon: an error bound

    Returns: word-topic matrix A
    """

    C, converged, total_iters, avg_steps, max_num_steps = exponentiated_gradients(Q, S)
    A = numpy.empty_like(C)

    denoms = [0] * A.shape[1]
    for k in range(len(denoms)):
        for iprime in range(A.shape[0]):
            prob = C[iprime][k] * priors[iprime]
            denoms[k] += prob
            A[iprime][k] = prob

    A = A / denoms
    return A, C


def top_words(A, num_words=20):
    """Finds the top words from each topic in A.

    Inputs:
        A: The word-topic matrix
        num_words: the number of top words to find

    Returns: A matrix of the indices of the top words
    from each topic (column) of A.
    """
    A = numpy.transpose(A)

    tops = [(0, 0.0)]*A.shape[0]

    for topic in range(A.shape[0]):
        # get the indices of the top values
        tops[topic] = list(sorted(enumerate(A[topic]), key=lambda x: x[1], reverse=True))[:num_words]
    return tops


def calculate_coherences(top_words, term_by_doc):
    """Calculates the semantic coherence scores of the topics based on their top words.

    Inputs:
        top_words: A list where each element is a list of the index and value of the top
                   words in that topic
        term_by_doc: The term-by-document matrix H of word counts

    Returns: A list of doubles representing the coherence score of the topics
    """
    coherences = [0.0] * len(top_words)
    columns = numpy.transpose(term_by_doc)

    for topic in range(len(top_words)):
        for word_1 in range(1, len(top_words[topic])):
            for word_2 in range(word_1):
                top_word_1 = top_words[topic][word_1][0]
                top_word_2 = top_words[topic][word_2][0]
                co_freq = len(list(filter(lambda x: x[top_word_1] >= 1 and x[top_word_2] >= 1, columns)))
                freq = len(list(filter(lambda x: x[top_word_2] >= 1, columns)))
                coherences[topic] += log((co_freq + 1) / freq)
    return coherences


# TODO: docstrings
# Currently both merge and mate use averaging
def merge(S, topic1, topic2):
    """Adds a new anchor to S that is a combination of topic1 and topic2, removing both"""
    maxsize = S.shape[0]
    assert topic1 < maxsize and topic2 < maxsize
    merged = _merge_average(S[topic1], S[topic2])
    S = numpy.append(numpy.delete(S, [topic1, topic2], axis=0), [merged], axis=0)
    return S


def mate(S, topic1, topic2):
    """Adds a new anchor to S that is a combination of topic1 and topic2, leaving all 3"""
    maxsize = S.shape[0]
    assert topic1 < maxsize and topic2 < maxsize
    merged = _merge_average(S[topic1], S[topic2])
    S = numpy.append(S, [merged], axis=0)
    return S


def _merge_average(v1, v2):
    """Helper function that merges two distributions"""
    return (v1 + v2) / 2

def _merge_multiply(v1, v2):
    """Helper function that merges two distributions"""
    tmp = numpy.multiply(v1, v2)
    # renormalize
    return tmp / numpy.sum(tmp)


def profile_exponentiated(convex=None, num_trials=1, target_size = 1000, c1=[10**i for i in range(-17, -13)],
                          noise=[i/1000 for i in range(10, 11, 5)], epsilons=[10**i for i in range(-5, -10, -2)],
                          decays=[i*.01 for i in range(10, 100, 30)],
                          filename="exponentiated_results.txt"):

    if convex is None:
        convex = numpy.random.uniform(size=(20, 1000))  # simulates 20 topics and 1000-length vocab
        sums = numpy.sum(convex, axis=1, keepdims=True)
        convex /= sums

    with open(filename, 'w') as results:
        initial_time = time.time()
        for i in range(num_trials):
            # Pick weights uniformly at random for now - could use a Dirichlet for a better model in the future
            target_weights = numpy.random.uniform(size=(target_size, convex.shape[0]))

            # Add an anchor to each topic to better fit our model
            for j in range(convex.shape[0]):
                singular = numpy.zeros(convex.shape[0])
                singular[j] = 1
                target_weights[j] = singular

            # Row normalize our weights to fit on the simplex
            sums = numpy.sum(target_weights, axis=1, keepdims=True)
            target_weights /= sums

            results.write("\n\nNew target: \n")

            # Add some noise to better model data
            for error_percent in noise:
                results.write("\n")
                if error_percent > 0.0:
                    error = numpy.random.normal(loc=0.0, scale=error_percent*(1/convex.shape[0]),
                                                size=(target_size, convex.shape[1]))
                else:
                    error = numpy.zeros((target_size, convex.shape[1]))

                target_matrix = numpy.dot(target_weights, convex) + error
                min_target = numpy.min(target_matrix)

                # Make sure we don't end up with any negative entries due to error
                if min_target < 0.0:
                    target_matrix += min_target

                results.write("\n")

                for const in c1:
                    results.write("\n")
                    for epsilon in epsilons:
                        results.write("\n")
                        for decay in decays:
                            start = time.time()

                            reconstructed, converged, total_iters, avg_steps, max_num_steps = \
                                exponentiated_gradients(target_matrix, convex, epsilon, c1=const, decay=decay)
                            taken = time.time() - start
                            mean_error = numpy.mean(numpy.linalg.norm(target_weights - reconstructed, ord=2, axis=1))

                            print("noise: %.3f\tc1: %.1f\tdecay: %.2f\tepsilon: %.7f\terror: %.4f\tsecs: %d\tconverged: %d/%d\n\t"
                                          "iters: %d\tavg steps: %.1f\tmax steps: %d\n" %
                                          (error_percent, log(const, 10), decay, log(epsilon, 10), mean_error*1000, taken, converged, target_size,
                                           total_iters, avg_steps, max_num_steps))
                            # print("Completed a reconstruction in: %d seconds. Noise: %.2f, Converged: %d/%d, Average steps: %.1f" %
                            #       (taken, error_percent, converged, target_size, avg_steps))
                            results.write("noise: %.3f\tc1: %.1f\tdecay: %.2f\tepsilon: %.7f\terror: %.4f\tmillis: %d\tconverged: %d/%d\n\t"
                                          "iters: %d\tavg steps: %.1f\tmax steps: %d\n" %
                                          (error_percent, log(const, 10), decay, log(epsilon, 10), mean_error*1000, taken*1000, converged, target_size,
                                          total_iters, avg_steps, max_num_steps))
            print("Completed %d reconstructions with %d rows each! Total time: %d seconds!" %
                  (len(noise)*len(c1) * len(epsilons) * len(decays), target_size, time.time()-initial_time))

        results.write("\nReconstructed %d rows in %d seconds" % (num_trials*len(epsilons)*len(c1),
                                                                 time.time()-initial_time))


if __name__ == '__main__':
    profile_exponentiated(filename="exponentiated_results_1000.txt")
