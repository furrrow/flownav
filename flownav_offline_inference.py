import argparse
import os
import time

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
matplotlib.use("TkAgg")
import yaml

import pickle
from PIL import Image as PILImage
import argparse
import torchdiffeq
from pathlib import Path
# Custom Imports
from flownav.training.utils import get_action
from flownav.visualizing.plot import plot_trajs_and_points
from deployment.src.utils_offline import (to_numpy, transform_images, load_model,
                                          load_calibration, overlay_path)
from deployment.src.utils_offline import RGB_color_dict as color_dict
"""
offline_inference.py
custom inference script to test out flownav,
a combination of code from train.py and navigate.py
"""

# CONSTANTS
TOPOMAP_IMAGES_DIR = "/home/jim/Projects/prune/deployment/topomaps/images"
CAMERA_MATRIX_DIR = "/home/jim/Projects/prune/deployment/camera_matrix.json"
ROBOT_CONFIG_PATH ="/home/jim/Projects/prune/deployment/config/ghost.yaml"
MODEL_CONFIG_PATH = "./deployment/config/models.yaml"
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"]

def main(config: dict) -> None:
    # Set up the device
    if torch.cuda.is_available():
        gpu_id = "0"
    device = torch.device(
        f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    exp_dir = args.exp_dir
    os.makedirs(exp_dir, exist_ok=True)

    k_steps = args.k_steps

    ckpt_path = Path(args.ckpt)
    cur_exp_dir = f"{exp_dir}/{args.model}_{ckpt_path.name}_{args.dir}_{args.goal_node}_{args.k_steps}"
    os.makedirs(cur_exp_dir, exist_ok=True)

    cur_exp_im_dir = f"{cur_exp_dir}/images"
    os.makedirs(cur_exp_im_dir, exist_ok=True)

    cur_exp_pkl_dir = f"{cur_exp_dir}/pkl"
    os.makedirs(cur_exp_pkl_dir, exist_ok=True)

    # load model parameters
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    context_size = model_params["context_size"]

    # load model weights
    ckpth_path = args.ckpt
    if os.path.exists(ckpth_path):
        print(f"Loading model from {ckpth_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
    model = load_model(
        ckpth_path,
        model_params,
        device,
    )
    model = model.to(device)
    model.eval()

    # load topomap
    topomap_filenames = sorted(os.listdir(os.path.join(
        TOPOMAP_IMAGES_DIR, args.dir)), key=lambda x: int(x.split(".")[0]))
    topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{args.dir}"
    num_nodes = len(os.listdir(topomap_dir))
    topomap = []
    for i in range(num_nodes):
        image_path = os.path.join(topomap_dir, topomap_filenames[i])
        topomap.append(PILImage.open(image_path))

    context_size = model_params["context_size"]
    context_queue = topomap[:context_size+1]
    # if len(context_queue) < self.context_size + 1:
    #     self.context_queue.append(self.obs_img)
    # else:
    #     self.context_queue.pop(0)
    #     self.context_queue.append(self.obs_img)

    cam_matrix, dist_coeffs, T_base_from_cam = load_calibration(CAMERA_MATRIX_DIR)
    T_cam_from_base = np.linalg.inv(T_base_from_cam)
    # run_navigation_loop, once.
    chosen_waypoint = np.zeros(4)
    obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
    obs_images = torch.split(obs_images, 3, dim=1)
    obs_images = torch.cat(obs_images, dim=1)
    obs_images = obs_images.to(device) # [1, 15, 96, 96]
    # Definition of the goal mask (convention: 0 = no mask, 1 = mask)
    one_mask = torch.ones(1).long().to(device)
    no_mask = torch.zeros(1).long().to(device)

    closest_node = 0
    assert -1 <= args.goal_node < len(topomap), "Invalid goal index"
    if args.goal_node == -1:
        goal_node = len(topomap) - 1
    else:
        goal_node = args.goal_node
    start = max(closest_node - args.radius, 0)
    end = min(closest_node + args.radius + 1, goal_node)
    goal_image = [transform_images(g_img, model_params["image_size"], center_crop=False).to(device) for g_img in
                  topomap[start:end + 1]]
    goal_image = torch.concat(goal_image, dim=0)

    # navigation
    obsgoal_cond = model('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1),
                              goal_img=goal_image, input_goal_mask=no_mask.repeat(len(goal_image)))
    dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
    dists = to_numpy(dists.flatten())

    # exploration
    obs_cond = model('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1),
                         goal_img=goal_image, input_goal_mask=one_mask.repeat(len(goal_image)))
    min_idx = np.argmin(dists)
    closest_node = min_idx + start

    # infer action
    with torch.no_grad():
        start_time = time.time()
        obs_cond = obs_cond[
            min(min_idx + int(dists[min_idx] < args.close_threshold), len(obsgoal_cond) - 1)].unsqueeze(0)
        obsgoal_cond = obsgoal_cond[
            min(min_idx + int(dists[min_idx] < args.close_threshold), len(obsgoal_cond) - 1)].unsqueeze(0)

        if len(obs_cond.shape) == 2:
            obs_cond = obs_cond.repeat(args.num_samples, 1)
            obsgoal_cond = obsgoal_cond.repeat(args.num_samples, 1)
        else:
            obs_cond = obs_cond.repeat(args.num_samples, 1, 1)
            obsgoal_cond = obsgoal_cond.repeat(args.num_samples, 1, 1)

        # Exploration
        output = torch.randn((len(obs_cond), model_params["len_traj_pred"], 2), device=device)
        traj = torchdiffeq.odeint(
            lambda t, x: model.forward(
                "noise_pred_net", sample=x, timestep=t, global_cond=obs_cond
            ),
            output,
            torch.linspace(0, 1, 10, device=device),
            atol=1e-4,
            rtol=1e-4,
            method="euler",
        )
        uc_actions = to_numpy(get_action(traj[-1]))

        # Navigation
        noisy_action = torch.randn(
            (args.num_samples, model_params["len_traj_pred"], 2), device=device)

        traj = torchdiffeq.odeint(
            lambda t, x: model.forward("noise_pred_net", sample=x, timestep=t, global_cond=obsgoal_cond),
            noisy_action,
            torch.linspace(0, 1, k_steps, device=device),
            atol=1e-4,
            rtol=1e-4,
            method="euler",
        ) # torch.Size([k_steps, 8, 8, 2])
        gc_actions = to_numpy(get_action(traj[-1]))  # [8, 8, 2]

        proc_time = time.time() - start_time
        mean_proc_time = proc_time / noisy_action.shape[0]
        print("Mean Processing Time UC", mean_proc_time)
        print("Processing Time UC", proc_time)

        # sampled_actions_msg = Float32MultiArray()
        message_data = np.concatenate((np.array([0]), gc_actions.flatten()))
        # sampled_actions_msg.data = message_data.tolist()
        # sampled_actions_pub.publish(sampled_actions_msg)
        # print("sampled_actions_msg", message_data)
        current_action = gc_actions[0]
        chosen_waypoint = current_action[args.waypoint]

    # plot distribution:
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 3, figure=fig)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[:, 1:])

    fig.suptitle("green goal, red explore, magenta best_path")
    uc_actions = list(uc_actions)
    gc_actions = list(gc_actions)
    action_label = gc_actions[0]
    traj_list = np.concatenate(
        [
            uc_actions,
            gc_actions,
            action_label[None],
        ],
        axis=0,
    ) # [17,8,2]
    traj_colors = (
            ["red"] * len(uc_actions) + ["green"] * len(gc_actions) + ["magenta"]
    )
    mock_goal_pos = np.array([10, 0])
    traj_alphas = [0.1] * (len(uc_actions) + len(gc_actions)) + [1.0]
    point_list = [np.array([0, 0]), torch.Tensor(mock_goal_pos)]
    point_colors = ["green", "red"]
    point_alphas = [1.0, 1.0]
    plot_trajs_and_points(
        ax=ax00,
        list_trajs=traj_list,
        list_points=point_list,
        traj_colors=traj_colors,
        point_colors=point_colors,
        traj_labels=None,
        point_labels=None,
        quiver_freq=0,
        traj_alphas=traj_alphas,
        point_alphas=point_alphas,
    )
    obs_image = np.array(context_queue[0])
    goal_image = np.array(topomap[-1])
    # obs_image = np.moveaxis(obs_image, 0, -1)
    # goal_image = np.moveaxis(goal_image, 0, -1)
    obs_image = overlay_path(np.array(gc_actions[1:]), obs_image, cam_matrix, T_cam_from_base, color_dict['GREEN'])
    obs_image = overlay_path(np.array(gc_actions[0]), obs_image, cam_matrix, T_cam_from_base, color_dict['BLUE'])
    ax11.imshow(obs_image)
    ax00.set_title("action predictions")
    ax11.set_title("observation")

    ax01.imshow(goal_image)
    ax01.set_title(f"goal")
    plt.show()


    # waypoint_msg = Float32MultiArray()
    waypoint_msg = chosen_waypoint.flatten().tolist()
    # waypoint_msg.data = chosen_waypoint.flatten().tolist()
    # waypoint_pub.publish(waypoint_msg)
    print("goal reached message", waypoint_msg)

    print(f"CHOSEN WAYPOINT: {chosen_waypoint}")

    reached_goal = closest_node == goal_node
    # goal_reached_msg = Bool()
    # goal_reached_msg.data = bool(reached_goal)
    # goal_pub.publish(goal_reached_msg)
    print("goal reached message")

    if reached_goal:
        print("Reached goal! Stopping...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline script to run flownav")
    # Parse command line arguments
    parser.add_argument(
        "--model",
        "-m",
        default="flownav",
        type=str,
        help="Model to run: Only FlowNav is supported currently (default: flownav)",
    )
    parser.add_argument(
        "--ckpt",
        default="./weights/flownav_weights.pth",
        type=str,
        help="Checkpoint path",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2,
        type=int,
        help="index of the waypoint used for navigation (default: 2)",
    )
    parser.add_argument(
        "--k_steps",
        "-k",
        default=10,
        type=int,
        help="Number of time steps",
    )
    parser.add_argument(
        "--dir",
        "-topo_dir",
        default="antonov",
        type=str,
        help="path to topomap images",
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="goal node index in the topomap (default: -1)",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="temporal distance within the next node in the topomap before localizing to it",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=4,
        type=int,
        help="temporal number of locobal nodes to look at in the topopmap for localization",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help="Number of actions sampled from the exploration model",
    )
    parser.add_argument(
        "--exp_dir",
        "-d",
        default="./nav_experiments",
        type=str,
        help="Path to log experiment results",
    )

    args = parser.parse_args()
    main(args)

