import utility_functions_5 as util
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.linalg import inv
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import  mahalanobis


def init_tracker(graph, init_cov):
    # INPUT: Graph structure
    # OUTPUT: Initialized tracks for binary mask sequence
    # Structure of track object:
    #   Dictionary of detection information
    #   One track: List of dictionaries
    #   All tracks: List of list of dictionaries

    all_tracks = []
    # Create a dictionary for each detection in the first frame
    for i in range(0, len(graph.img_nodes[0])):
        # Each dictionary stores frame index, state information, number of misdetections
        # Each state is [pos_X, pos_Y, vel_X, vel_Y, cov]
        node_ind = graph.img_nodes[0][i]
        node = graph.G.node[node_ind]
        state = np.array([node['centroid'][0], node['centroid'][1], 0, 0])
        dict = {"frame": 0, "global_ind": node_ind, "state": state, "md": 0, "cov": init_cov}
        one_track = [dict]
        all_tracks.append(one_track)
    return all_tracks


def kalman_predict(tracks, frame_num):
    # INPUT: All tracks
    # OUTPUT: Predicted outputs
    dt = 0.1
    PI = np.array([[5, 0], [0, 5]])
    A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    V = np.array([[dt, 0], [0, dt], [1, 0], [0, 1]])
    Q = np.diag([10, 10, 5, 5])
    max_missed_det = 3

    # For every track input, predict and append to the track list
    for i in range(0, len(tracks)):
        state = tracks[i][-1]['state']
        missed_det = tracks[i][-1]['md']
        cov = np.asarray(tracks[i][-1]['cov'])

        # Predict state only if number of missed detections is less than maximum
        if missed_det < max_missed_det:
            # Predict next state
            pred_state = np.dot(A, state)

            # Predict next covariance
            # pred_cov = A*cov*A.T + Q
            pred_cov = np.dot(A, np.dot(cov, A.T)) + np.dot(V, np.dot(PI, V.T))

            # Make a dictionary with prediction values
            dict = {"frame": frame_num, "state": pred_state, "md": 0, "cov": pred_cov}

            # Append to track
            tracks[i].append(dict)
    return


def hungarian_algorithm(graph, all_tracks, frame_num, init_cov):
    # INPUT: Current tracks
    # OUTPUT: Updated track assignments
    node_inds = graph.img_nodes[frame_num]
    num_detections = len(node_inds)
    num_tracks = len(all_tracks)
    max_md = 3
    max_dist = 80

    cost = []
    cost_to_track = []
    assignments = -1*np.ones(num_tracks, dtype=int)
    det_check = -1*np.ones(num_detections, dtype=int)

    # Construct the cost matrix with Mahalanobis distance
    for t in range(0, num_tracks):
        if all_tracks[t][-1]['md'] < max_md:
            track_cost = np.zeros(num_detections)
            for d in range(0, num_detections):
                node = graph.G.node[node_inds[d]]
                track_state = all_tracks[t][-1]['state'][0:2]
                track_cov = all_tracks[t][-1]['cov'][0:2, 0:2]
                # track_cost[d] = mahalanobis(node['centroid'], track_state, track_cov)
                track_cost[d] = np.linalg.norm(node['centroid'] - track_state)
            # Append costs to list only if minimum cost in track is lesser than max_dist
            if np.min(track_cost) < max_dist:
                cost_to_track.append(t)
                cost.append(track_cost)
            else:
                assignments[t] = -2
                if len(all_tracks[t]) == 1:
                    all_tracks[t][0]['md'] = 1
                else:
                    all_tracks[t][-1]['md'] = all_tracks[t][-2]['md'] + 1
        else:
            assignments[t] = -2
    cost = np.array(cost)

    # Find lowest assignment pairs with the Hungarian algorithm
    [row_ind, col_ind] = linear_sum_assignment(cost)
    assignments[np.array(cost_to_track)[row_ind]] = col_ind

    # Handling extra detections by starting new tracks
    det_check[col_ind] = 1
    extra_det = np.where(det_check != 1)
    for e_ind in range(0, len(extra_det[0])):
        e = extra_det[0][e_ind]
        node_ind = graph.img_nodes[frame_num][e]
        node = graph.G.node[node_ind]
        state = np.array([node['centroid'][0], node['centroid'][1], 0, 0])
        dict = {"frame": frame_num, "global_ind": node_ind, "state": state, "md": 0, "cov": init_cov}
        all_tracks.append([dict])
        assignments = np.concatenate([assignments, [e]], axis=0)

    # Handling extra tracks
    extra_track = np.where(assignments == -1)
    for e in range(0, len(extra_track[0])):
        track_ind = extra_track[0][e]
        # Find nearest centroids
        cost_ind = np.where(np.array(cost_to_track) == track_ind)[0][0]
        min_cost = np.argmin(cost[cost_ind])
        assignments[track_ind] = min_cost
        all_tracks[track_ind][-1]['md'] = 0
    return assignments


def kalman_update(graph, all_tracks, assignments, frame_num):
    # INPUT: Current tracks and Hungarian algorithm assignments
    # OUTPUT: Updated tracks

    # Initialize params
    R = np.diag([5, 5])
    H = np.zeros([2, 4])
    H[0, 0] = 1
    H[1, 1] = 1

    # Loop through tracks
    for t in range(0, len(all_tracks)):
        # Check if detection was found
        if assignments[t] >= 0:
            det = assignments[t]
            track = all_tracks[t][-1]
            # Get detection centroid
            node_ind = graph.img_nodes[frame_num][det]
            node_centroid = graph.G.node[node_ind]['centroid']
            # Update graph
            track['global_ind'] = node_ind
            if len(all_tracks[t]) > 1:
                md = all_tracks[t][-2]['md']
                prev_track = all_tracks[t][-2-md]
                prev_node_ind = prev_track['global_ind']
                graph.G.add_edge(prev_node_ind, node_ind)

            # Apply measurement step update
            temp = np.dot(H, np.dot(track['cov'], H.T)) + R
            K = np.dot(track['cov'], np.dot(H.T, inv(temp)))
            innovation = node_centroid - np.dot(H, track['state'])
            track['state'] += np.dot(K, innovation)
            track['cov'] = np.dot((np.eye(4) - np.dot(K, H)), track['cov'])

    return

