import utility_functions_5 as util
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import cv2


class CellTracks:

    # Constructor - calculates features for each detection and initializes graphical model
    def __init__(self, bin_mask, vid_img, search_dist, min_area):
        self.search_dist = search_dist
        self.bin_mask = bin_mask
        self.vid_img = vid_img
        # Calculate vector of features for each image
        feats, self.contours, centroid_vect, self.mean_area = util.gen_features(bin_mask, min_area)
        print "Calculated feature vector for all nodes..."
        # Initialize nodes in graph
        self.G = nx.DiGraph()
        # Add a node for each detection
        self.img_nodes = []
        for i in range(0, len(feats)):
            curr_nodes = []
            for d in range(0, feats[i].shape[0]):
                # centroid = feats[i][0:2, d]
                feats_node = feats[i][d, :]
                centroid_node = centroid_vect[i][d]
                curr_nodes.append(self.G.number_of_nodes())
                self.G.add_node(self.G.number_of_nodes(), feats=feats_node, centroid=centroid_node)
            self.img_nodes.append(curr_nodes)
        self.num_frames = len(self.img_nodes)
        print "Initialized graphical model..."

    # Draws DAG representation of cell tracks
    # Inputs:
    #   with_labels - true if the graph should also draw cell #'s
    def draw_graph(self, with_labels):
        pos = {}
        node_count = 0
        for i in range(0, len(self.img_nodes)):
            for d in range(0, len(self.img_nodes[i])):
                pos[node_count] = ((i + 1), (d + 1))
                node_count += 1
        nx.draw(self.G, pos, with_labels=with_labels)
        plt.show()

    # Draws bounding box around specified contour
    #   contour - contour being bounded
    #   label   - cell/detection number to draw on box
    #   frame   - frame in which to draw box
    def draw_box(self, contour, label, frame, img):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "#" + format(label), (x + w + 10, y + h), 2, 0.4, (0, 255, 0))
        cv2.putText(img, "#" + format(label), (x - 10, y - 5), 2, 0.4, (0, 255, 0))
        return img

    # Draws the cell tracks as labeled bounding boxes in video sequence
    def gen_images(self):
        # Loop through frames
        track_count = 0
        nodes_drawn = np.zeros([self.G.number_of_nodes()])
        for frame in range(0, len(self.img_nodes)):
            img = self.bin_mask[frame].copy()
            # Loop through detections in current frame
            for i in range(0, len(self.img_nodes[frame])):
                detection = self.img_nodes[frame][i]
                # Check for predecessors with a track label
                in_edges = list(self.G.in_edges(detection))
                if len(in_edges) == 1:
                    # Inherit parent label
                    parent = in_edges[0][0]
                    nodes_drawn[detection] = nodes_drawn[parent]
                elif len(in_edges) == 0:
                    nodes_drawn[detection] = track_count
                    track_count += 1
                else:
                    print "ERROR TOO MANY INCOMING EDGES"
                # Draw bounding box
                track_label = nodes_drawn[detection]
                contour = self.contours[frame][i]
                img = self.draw_box(contour, track_label, frame, img)
            cv2.imwrite("./output/" + '{:03d}'.format(frame) + ".png", img)
        return

    # Draws the cell tracks as labeled bounding boxes in video sequence
    def gen_images_reverse(self, out_path):
        # Loop through frames
        track_count = 0
        nodes_drawn = np.zeros([self.G.number_of_nodes()])
        for frame in range(len(self.img_nodes)-1, -1, -1):
            img = self.bin_mask[frame].copy()
            # Loop through detections in current frame
            for i in range(0, len(self.img_nodes[frame])):
                detection = self.img_nodes[frame][i]
                # Check for predecessors with a track label
                out_edges = list(self.G.out_edges(detection))
                if len(out_edges) == 1:
                    # Inherit parent label
                    parent = out_edges[0][1]
                    nodes_drawn[detection] = nodes_drawn[parent]
                elif len(out_edges) == 0:
                    nodes_drawn[detection] = track_count
                    track_count += 1
                else:
                    nodes_drawn[detection] = track_count
                    track_count += 1
                # Draw bounding box
                track_label = nodes_drawn[detection]
                contour = self.contours[frame][i]
                img = self.draw_box(contour, track_label, frame, img)
            cv2.imwrite(out_path + '{:03d}'.format(frame) + ".png", img)
        return

    def draw_tracks(self, tracks):
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255),
                  (255, 127, 255), (127, 0, 255), (127, 0, 127)]
        track_frame_ind = np.zeros(len(tracks), dtype=int)
        track_paths = [[] for x in xrange(len(tracks))]
        for frame in range(0, len(self.img_nodes)):
            img = cv2.cvtColor(self.bin_mask[frame].copy(), cv2.COLOR_GRAY2RGB)
            col_img = self.vid_img[frame].copy()
            # Find tracks in current frame
            for track_ind in range(0, len(tracks)):
                last_ind = track_frame_ind[track_ind]
                if last_ind < len(tracks[track_ind]) and tracks[track_ind][last_ind]['frame'] == frame:
                    # Add position to path
                    pos = tracks[track_ind][last_ind]['state'][0:2].astype(int)
                    track_paths[track_ind].append(pos)
                    if len(track_paths[track_ind]) > 10:
                        del track_paths[track_ind][0]
                    # Draw track
                    prev_pt = track_paths[track_ind][0]
                    if len(track_paths[track_ind]) > 1:
                        for pt in track_paths[track_ind][1:]:
                            # Draw line from prev point to current
                            cv2.line(img, (prev_pt[0], prev_pt[1]), (pt[0], pt[1]), colors[track_ind % 9], 2)
                            cv2.line(col_img, (prev_pt[0], prev_pt[1]), (pt[0], pt[1]), colors[track_ind % 9], 2)
                            # Update prev point
                            prev_pt = pt
                    # Increment frame count for that track
                    track_frame_ind[track_ind] += 1
            cv2.imwrite("./output/" + '{:03d}'.format(frame) + ".png", np.hstack((img, col_img)))
        return
