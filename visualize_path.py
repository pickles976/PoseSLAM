import numpy as np
import matplotlib.pyplot as plt
from lib.visualization import plotting
import os

def load_poses(filepath):
    """
    Loads the GT poses

    Parameters
    ----------
    filepath (str): The file path to the poses file

    Returns
    -------
    poses (ndarray): The GT poses
    """
    poses = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            T = np.fromstring(line, dtype=np.float64, sep=' ')
            T = T.reshape(3, 4)
            T = np.vstack((T, [0, 0, 0, 1]))
            poses.append(T)
    return poses

poses = load_poses('data_odometry_poses/dataset/poses/00.txt')

estimated_path = []
for pose in poses:
    estimated_path.append((pose[0, 3], pose[2, 3]))

plotting.visualize_paths(estimated_path, estimated_path, "Visual Odometry", file_out=os.path.basename("data_dir") + ".html")