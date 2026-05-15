import argparse
import os
import time

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
from inference_point_based import RewardInferenceRunner
from frechetdist import frdist
from dtaidistance import dtw, dtw_ndim
"""
offline_inference.py
custom inference script to test out flownav,
a combination of code from train.py and navigate.py
"""

# CONSTANTS
TOPOMAP_IMAGES_DIR = "/home/jim/Projects/prune/deployment/topomaps/images"
CAMERA_MATRIX_DIR = "/home/jim/Projects/prune/deployment/camera_matrix.json"
# TOPOMAP_IMAGES_DIR = "/workspace/prune/deployment/topomaps/images"
# CAMERA_MATRIX_DIR = "/workspace/prune/deployment/camera_matrix.json"
ROBOT_CONFIG_PATH ="./deployment/config/robot.yaml"
MODEL_CONFIG_PATH = "./deployment/config/models.yaml"


def make_video_writer(video_path: str, fps: float):
    suffix = Path(video_path).suffix.lower()
    if suffix == ".gif":
        return animation.PillowWriter(fps=fps), video_path

    if animation.writers.is_available("ffmpeg"):
        return animation.FFMpegWriter(fps=fps), video_path

    fallback_path = str(Path(video_path).with_suffix(".gif"))
    print(f"ffmpeg is not available; saving GIF instead: {fallback_path}")
    return animation.PillowWriter(fps=fps), fallback_path

def resample_path_2d(path: np.ndarray, k: int) -> np.ndarray:
    """
    Evenly resample a sequence of 2D points to length k using linear interpolation.
    Expects ``path`` shape (n, 2); returns float32 array shape (k, 2).
    """
    if path.size == 0:
        return np.zeros((k, 2), dtype=np.float32)
    if len(path) == 1:
        return np.repeat(path, k, axis=0)
    deltas = path[1:] - path[:-1]
    seg_len = np.linalg.norm(deltas, axis=1)
    cum = np.concatenate([np.array([0.0], dtype=np.float32), np.cumsum(seg_len, dtype=np.float32)])
    total = cum[-1]
    if total == 0:
        return np.repeat(path[:1], k, axis=0)

    target = np.linspace(0.0, float(total), num=k, dtype=np.float32)
    out = np.empty((k, path.shape[1]), dtype=np.float32)
    for i, t in enumerate(target):
        j = np.searchsorted(cum, t, side="right") - 1
        j = int(np.clip(j, 0, len(seg_len) - 1))
        t0, t1 = cum[j], cum[j + 1]
        alpha = 0.0 if t1 == t0 else float((t - t0) / (t1 - t0))
        out[i] = path[j] * (1 - alpha) + path[j + 1] * alpha
    return out

def prune_distance(points: np.ndarray, cutoff: float, k: int):
    """

    :param points: trajectories of xy points, (num_trajectories, traj_length, 2)
    :param cutoff: a distance measure
    :return: updated trajectory points
    """
    paths = np.array(points, dtype=np.float32) # (8, 8, 2)
    n_pts, pt_len, _ = paths.shape
    first_points = np.expand_dims(paths[:, 0, :], 1).repeat(pt_len, axis=1)
    deltas = paths - first_points
    seg_len = np.linalg.norm(deltas, axis=-1)
    path_check = seg_len > cutoff
    for i, i_path in enumerate(paths):
        if not True in path_check[i]:
            continue
        sub_path = i_path[:np.argmax(path_check[i])]
        paths[i] = resample_path_2d(sub_path, k)
    return paths

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
    distance_cutoff = 4.5 # won't consider paths beyond this distance when doing steering
    ckpt_path = Path(args.ckpt)
    cur_exp_dir = f"{exp_dir}/{args.model}_{ckpt_path.name}_{args.dir}_{args.goal_node}_{args.k_steps}_{distance_cutoff}"
    os.makedirs(cur_exp_dir, exist_ok=True)

    cur_exp_im_dir = f"{cur_exp_dir}/images"
    os.makedirs(cur_exp_im_dir, exist_ok=True)

    cur_exp_pkl_dir = f"{cur_exp_dir}/pkl"
    os.makedirs(cur_exp_pkl_dir, exist_ok=True)

    video_path = args.video_path
    if video_path is None:
        video_path = os.path.join(cur_exp_dir, "navigation.mp4")
    video_parent = os.path.dirname(video_path)
    if video_parent:
        os.makedirs(video_parent, exist_ok=True)
    video_writer, video_path = make_video_writer(video_path, args.video_fps)

    with open(ROBOT_CONFIG_PATH, "r") as f:
        robot_config = yaml.safe_load(f)
    robot_config = robot_config[args.robot]
    IMG_SIZE = (robot_config["img_w"], robot_config["img_h"]) # (1280, 720)

    if args.mode == "navigate":
        mode = "navigate"
    else:
        mode = "explore"

    if args.steer:
        # Reward model
        rm_ckpt_path = "./weights/epoch_029.pt"
        rm_config_path = "/home/jim/Projects/prune/config/config_point_based.yaml"
        reward_runner = RewardInferenceRunner(checkpoint_path=rm_ckpt_path, config_path=rm_config_path, verbose=True)
    # load model parameters
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

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
    rewards = None
    # load topomap
    topomap_filenames = sorted(os.listdir(os.path.join(
        TOPOMAP_IMAGES_DIR, args.dir)), key=lambda x: int(x.split(".")[0]))
    topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{args.dir}"
    num_nodes = len(os.listdir(topomap_dir))
    topomap = []
    for i in range(num_nodes):
        image_path = os.path.join(topomap_dir, topomap_filenames[i])
        topomap_img = PILImage.open(image_path)
        if topomap_img.size != IMG_SIZE:
            topomap_img = topomap_img.resize(IMG_SIZE)
        topomap.append(topomap_img)

    cam_matrix, dist_coeffs, T_base_from_cam = load_calibration(CAMERA_MATRIX_DIR)
    T_cam_from_base = np.linalg.inv(T_base_from_cam)

    # Definition of the goal mask (convention: 0 = no mask, 1 = mask)
    one_mask = torch.ones(1).long().to(device)
    no_mask = torch.zeros(1).long().to(device)

    assert -1 <= args.goal_node < len(topomap), "Invalid goal index"
    if args.goal_node == -1:
        goal_node = len(topomap) - 1
    else:
        goal_node = args.goal_node

    context_size = model_params["context_size"]
    last_start_img = len(topomap) - context_size - 1
    if args.start_img > last_start_img:
        raise ValueError(
            f"start_img={args.start_img} does not leave enough images for "
            f"context_size={context_size}; last valid start is {last_start_img}"
        )

    closest_node = args.start_img
    plt.ion()
    fig = plt.figure(figsize=(16, 8))
    video_writer.setup(fig, video_path, dpi=args.video_dpi)
    for nav_idx, start_img in enumerate(range(args.start_img, last_start_img + 1)):
        context_queue = topomap[start_img:context_size + start_img + 1]
        rewards = None
        chosen_waypoint = np.zeros(4)
        print(f"\nNavigation iteration {nav_idx}: topomap {args.dir} window {start_img}-{start_img + context_size}")

        obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1)
        obs_images = obs_images.to(device)  # [1, 15, 96, 96]

        start = max(closest_node - args.radius, 0)
        end = min(closest_node + args.radius + 1, goal_node)
        goal_image = [transform_images(g_img, model_params["image_size"], center_crop=False).to(device) for g_img in
                      topomap[start:end + 1]]
        goal_image = torch.concat(goal_image, dim=0) # [6, 3, 96, 96]

        if mode == "explore":
            obs_cond = model('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1),
                             goal_img=goal_image, input_goal_mask=one_mask.repeat(len(goal_image)))
            dists = model("dist_pred_net", obsgoal_cond=obs_cond)
        elif mode == "navigate":
            obs_cond = model('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1),
                                      goal_img=goal_image, input_goal_mask=no_mask.repeat(len(goal_image)))
            dists = model("dist_pred_net", obsgoal_cond=obs_cond)

        dists = to_numpy(dists.flatten())
        min_idx = np.argmin(dists)
        closest_node = min_idx + closest_node

        # infer action
        with torch.no_grad():
            start_time = time.time()
            if mode == "explore":
                obs_cond = obs_cond[
                    min(min_idx + int(dists[min_idx] < args.close_threshold), len(obs_cond) - 1)].unsqueeze(0)
            elif mode == "navigate":
                obs_cond = obs_cond[
                    min(min_idx + int(dists[min_idx] < args.close_threshold), len(obs_cond) - 1)].unsqueeze(0)

            if len(obs_cond.shape) == 2:
                obs_cond = obs_cond.repeat(args.num_samples, 1)
            else:
                obs_cond = obs_cond.repeat(args.num_samples, 1, 1)

            noisy_action = torch.randn(
                (args.num_samples, model_params["len_traj_pred"], 2), device=device)
            traj = torchdiffeq.odeint(
                lambda t, x: model.forward(
                    "noise_pred_net", sample=x, timestep=t, global_cond=obs_cond
                ),
                noisy_action,
                torch.linspace(0, 1, k_steps, device=device),
                atol=1e-4,
                rtol=1e-4,
                method="euler",
            )  # torch.Size([k_steps, 8, 8, 2])
            actions = to_numpy(get_action(traj[-1]))  # [8, 8, 2]
            # sampled_actions_msg = Float32MultiArray()
            message_data = np.concatenate((np.array([0]), actions.flatten()))
            # sampled_actions_msg.data = message_data.tolist()
            # sampled_actions_pub.publish(sampled_actions_msg)
            # print("sampled_actions_msg", message_data)
            current_action = actions[0]
            chosen_waypoint = current_action[args.waypoint]

            if args.steer:
                pruned_actions = prune_distance(actions, distance_cutoff, 8)
                image_tensor = torch.from_numpy(np.array(context_queue[-1])).permute(2, 0, 1).contiguous()  # (3, H, W)
                points_tensor = torch.from_numpy(pruned_actions)  # (M, K, 2)
                rewards = reward_runner.predict_rewards(image_tensor=image_tensor, points_tensor=points_tensor)
                best_action = torch.argmax(rewards).item()
                print("Predicted rewards:", rewards, "best reward action(red) :", best_action)
                # different distrance metric to make sure selected action does not veer too far:
                eval_dict = {}
                for idx, action in enumerate(pruned_actions):
                    eval_dict[idx] = {}
                    eval_dict[idx]["reward"] = rewards[idx].item()
                    eval_dict[idx]["frdist"]= frdist(action, pruned_actions[0])
                    eval_dict[idx]["dtw"] =  dtw_ndim.distance(action, pruned_actions[0])

            proc_time = time.time() - start_time
            mean_proc_time = proc_time / noisy_action.shape[0]
            print(f"Processing Time {proc_time:.4f} Mean Processing Time {mean_proc_time:.4f}")

        # plot distribution:
        fig.clf()
        gs = GridSpec(2, 3, figure=fig)
        ax00 = fig.add_subplot(gs[0, 0])
        ax01 = fig.add_subplot(gs[1, 0])
        ax11 = fig.add_subplot(gs[:, 1:])

        fig.suptitle(f"trajectory visualization with {args.model} | iteration {nav_idx}")
        actions = list(actions)
        traj_list = np.concatenate([actions], axis=0, )
        traj_list = traj_list[:, :, ::-1] # flip y-x for visualization purposes
        traj_list[:, :, 0] = -traj_list[:, :, 0] # flip x about 0 for visualization purposes
        traj_colors = ["blue"] + ["green"] * (len(actions)-1)
        traj_alphas = [0.75] + [0.1] * (len(actions)-1)
        if rewards is not None:
            new_traj_list = np.concatenate([pruned_actions], axis=0, )
            new_traj_list = new_traj_list[:, :, ::-1]  # flip y-x for visualization purposes
            new_traj_list[:, :, 0] = -new_traj_list[:, :, 0]  # flip x about 0 for visualization purposes
            traj_list = np.concatenate((traj_list, new_traj_list), axis=0, )
            new_colors = ["green"] * (len(pruned_actions))
            traj_colors[best_action] = "red"
            new_colors[best_action] = "red"
            traj_colors += new_colors
            traj_alphas += [0.5] * len(pruned_actions)

        point_list = [np.array([0, 0])]
        point_colors = ["green", "red"]
        point_alphas = [1.0, 1.0]
        plot_trajs_and_points(
            ax=ax01,
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
        obs_image = np.array(context_queue[-1])  # not sure which img is the best one to show...
        display_goal_image = np.array(topomap[closest_node])
        # obs_image = np.moveaxis(obs_image, 0, -1)
        # goal_image = np.moveaxis(goal_image, 0, -1)
        if args.steer:
            overlay_img = overlay_path(np.array(pruned_actions), obs_image, cam_matrix, T_cam_from_base, color_dict['GREEN'],
                                       metrics=eval_dict)
        else:
            overlay_img = overlay_path(np.array(actions), obs_image, cam_matrix, T_cam_from_base, color_dict['GREEN'])
        if overlay_img is not None:
            ax11.imshow(overlay_img)
        else:
            ax11.imshow(obs_image)

        ax00.imshow(display_goal_image)
        ax00.set_title(f"intermediate goal node {closest_node}")
        ax01.set_title("action predictions")
        ax11.set_title("observation, blue best path")

        fig.canvas.draw()
        fig.canvas.flush_events()
        image_path = os.path.join(cur_exp_im_dir, f"navigation_{nav_idx:04d}.png")
        fig.savefig(image_path, dpi=args.video_dpi, bbox_inches="tight")
        video_writer.grab_frame()
        # plt.pause(0.5)

        # waypoint_msg = Float32MultiArray()
        waypoint_msg = chosen_waypoint.flatten().tolist()
        # waypoint_msg.data = chosen_waypoint.flatten().tolist()
        # waypoint_pub.publish(waypoint_msg)
        # print("waypoint message", waypoint_msg)

        print(f"CHOSEN WAYPOINT: {chosen_waypoint}")

        reached_goal = closest_node == goal_node

        if reached_goal:
            print("Reached goal; saving and exiting.")
            break

    print(f"Finished {nav_idx + 1} navigation iterations.")
    video_writer.finish()
    print(f"Saved navigation video to {video_path}")
    plt.ioff()
    plt.show()


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
        default="iribe_corridoor",
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
        "--start-img",
        "-s",
        default=0,
        type=int,
        help="which topomap image to use as observation",
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
        default=10,
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
    parser.add_argument("-robo", "--robot", type=str, help="Robot Name",
                        default="ghost")
    parser.add_argument(
        "--mode",
        default="navigate",
        help="navigate or explore"
    )
    parser.add_argument(
        "--steer",
        action="store_true",
        help="whether to use the reward model steering"
    )
    parser.add_argument(
        "--video-path",
        default=None,
        type=str,
        help="path to save the navigation video; defaults to navigation.mp4 in the experiment directory",
    )
    parser.add_argument(
        "--video-fps",
        default=1.0,
        type=float,
        help="frames per second for the saved navigation video",
    )
    parser.add_argument(
        "--video-dpi",
        default=120,
        type=int,
        help="DPI used when encoding the saved navigation video",
    )

    args = parser.parse_args()
    main(args)
