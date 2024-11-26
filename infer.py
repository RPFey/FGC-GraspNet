""" Testing for GraspNet baseline model. """

import os
import sys
import numpy as np
import argparse
import time
from numba import jit
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from models.FGC_graspnet import FGC_graspnet
from models.loss import pred_decode

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='./grasp_data', help='Dataset root')
parser.add_argument('--checkpoint_path', default='/home/luyh/graspnet-baseline/logs_7155/best_noglobal/checkpoint.tar', help='Model checkpoint path')
parser.add_argument('--camera', default='realsense', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during inference [default: 1]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--num_workers', type=int, default=64, help='Number of workers used in evaluation [default: 30]')
cfgs = parser.parse_args()


# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

# Init the model
net = FGC_graspnet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
                      cylinder_radius=0.05, hmin=-0.02, hmax=0.02, is_training=False, is_demo=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)


end_points = {
    "point_clouds": torch.rand(2, 20000, 3).to(device)
}

pred = net(end_points)

import pdb; pdb.set_trace()
pass

