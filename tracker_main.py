# Import libraries
import utility_functions_5 as util
from CellTracks5_1 import CellTracks
import numpy as np
import pickle
from track_functions import *

# Set paths to data and to trained model
trial = 3
seg_path = "/home/davideh29/PycharmProjects/Cell_Tracker/normalized_frames/video_3/Binary/"
color_path = "/home/davideh29/PycharmProjects/Cell_Tracker/normalized_frames/video_3/Raw/"
out_path = "./output_" + format(trial) + "/"

# Set parameters
search_dist = 60    # Max distance moved per frame
min_area = 15

# Load color images and segmentations
bin_mask, vid_img = util.load_testing_data(seg_path, color_path)

# Get image dimensions
image_height = bin_mask.shape[1]
image_width = bin_mask.shape[2]

# Generate graphical model of detections
cell_tracks = CellTracks(bin_mask=bin_mask, vid_img=vid_img, search_dist=search_dist, min_area=min_area)

# Initialize tracker
init_covariance = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
tracks = init_tracker(graph=cell_tracks, init_cov=init_covariance)

# Loop till no more frames
for i in range(1, len(bin_mask)):
    if not i%10:
        print "Frame #" + format(i)

    # Kalman Filter step
    kalman_predict(tracks=tracks, frame_num=i)

    # Hungarian algorithm
    assignments = hungarian_algorithm(graph=cell_tracks, all_tracks=tracks, frame_num=i, init_cov=init_covariance)

    # Measurement step of Kalman Filter
    kalman_update(graph=cell_tracks, all_tracks=tracks, assignments=assignments, frame_num=i)

    # Visualization
    # cell_tracks.draw_graph(False)

# cell_tracks.draw_graph(False)
cell_tracks.draw_tracks(tracks)

print "Number of tracks: " + format(len(tracks))
