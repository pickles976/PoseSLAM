import numpy as np
import matplotlib.pyplot as plt
from lib.visualization import plotting
import os
from lib.util import load_poses

poses = load_poses('data_odometry_poses/dataset/poses/00.txt')

estimated_path = []
for pose in poses:
    estimated_path.append((pose[0, 3], pose[2, 3]))

plotting.visualize_paths(estimated_path, estimated_path, "Visual Odometry", file_out=os.path.basename("data_dir") + ".html")