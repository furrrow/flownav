import json
import math

# pytorch
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image as PILImage
from typing import List, Tuple, Dict, Optional

# models
from flownav.models.nomad import NoMaD, DenseNetwork
from flownav.models.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from flownav.data.data_utils import IMAGE_ASPECT_RATIO
import cv2

BGR_color_dict = { # BGR
    "RED" : (0, 0, 255),
    "GREEN" : (0, 255, 0),
    "BLUE" : (255, 0, 0),
    "CYAN" : (255, 255, 0),
    "YELLOW" : (0, 255, 255),
    "CUSTOM" : (125, 125, 125),
}

RGB_color_dict = { # RGB
    "RED" : (255, 0, 0),
    "GREEN" : (0, 255, 0),
    "BLUE" : (0, 0, 255),
    "CYAN" : (0, 255, 255),
    "YELLOW" : (255, 255, 0),
    "CUSTOM" : (125, 125, 125),
}

def load_model(
    model_path: str,
    config: dict,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Load a model from a checkpoint file (works with models trained on multiple GPUs)"""

    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=config["encoding_size"],
        context_size=config["context_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
        depth_cfg=config["depth"]
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)
    noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=config["encoding_size"],
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )
    dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
        
    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,
    )

    checkpoint = torch.load(model_path, map_location=device)

    state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    return model

def to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def transform_images(pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225]),
        ]
    )

    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))  # crop to the right ratio
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size) 
        transf_img = transform_type(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)
    

# clip angle between -pi and pi
def clip_angle(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi


def clip_angle(theta) -> float:
    """Clip angle to [-pi, pi]"""
    theta %= 2 * np.pi
    if -np.pi < theta < np.pi:
        return theta
    return theta - 2 * np.pi

def overlay_path(pts_cur: np.ndarray, img: Optional[np.ndarray] = None, cam_matrix: Optional[np.ndarray] = None,
                 T_cam_from_base: Optional[np.ndarray] = None,
                 path_color=(0, 0, 255), policy_color= RGB_color_dict['BLUE'], steer_color=RGB_color_dict['RED'],
                 metrics=None):
    if pts_cur.size == 0:
        print("pts_cur.size is zero...")
        return None
    if cam_matrix is None or T_cam_from_base is None:
        print("cam_matrix:", cam_matrix)
        print("T_cam_from_base:", T_cam_from_base)
        print("returning...")
        return None
    if img is None:
        print("img is none...")
        return None

    if len(pts_cur.shape) == 2:
        n_trajectories = 1
        pts_cur = np.expand_dims(pts_cur, 0)
    elif len(pts_cur.shape) == 3:
        n_trajectories = pts_cur.shape[0]
    else:
        print("error, unable to process pts_cur dimension", pts_cur.shape)
        return None
    metric_labels = {}
    if metrics is not None:
        reward_values = []
        for i in range(n_trajectories):
            metric = metrics.get(i, metrics.get(str(i), None))
            if metric is None:
                reward_values.append(np.nan)
                continue
            reward_values.append(metric.get("reward", np.nan))

            label_lines = [f"{i}"]
            if "reward" in metric:
                label_lines.append(f"rew {metric['reward']:.2f}")
            if "frdist" in metric:
                label_lines.append(f"frd {metric['frdist']:.2f}")
            if "dtw" in metric:
                label_lines.append(f"dtw {metric['dtw']:.2f}")
            metric_labels[i] = label_lines
        rewards = np.asarray(reward_values, dtype=np.float32)

    # Points in base frame -> camera frame -> pixels
    R_cb = T_cam_from_base[:3, :3]
    t_cb = T_cam_from_base[:3, 3]
    rvec, _ = cv2.Rodrigues(R_cb)
    overlay = img.copy()
    reward_labels = []
    for i in range(n_trajectories):
        pts_3d = np.hstack([pts_cur[i], np.zeros((pts_cur[i].shape[0], 1))])  # z=0 in base frame
        img_pts, _ = cv2.projectPoints(pts_3d, rvec, t_cb, cam_matrix, None)
        img_pts = img_pts.reshape(-1, 2)

        # Keep points in front of camera and inside image
        pts_cam = (R_cb @ pts_3d.T + t_cb.reshape(3, 1)).T
        valid_z = pts_cam[:, 2] > 0
        h, w = img.shape[:2]
        valid_xy = (
            (img_pts[:, 0] >= 0) & (img_pts[:, 0] < w) &
            (img_pts[:, 1] >= 0) & (img_pts[:, 1] < h)
        )
        keep = valid_z & valid_xy
        if not keep.any():
            print(f"out of {pts_cam.shape} points, no points kept in front of camera...")
            continue

        pts_pix = img_pts[keep].astype(int)
        my_color = path_color
        if i == 0:
            my_color = policy_color
        if metrics is not None:
            if i == np.argmax(rewards):
                my_color = steer_color
        if len(pts_pix) >= 2:
            cv2.polylines(overlay, [pts_pix], isClosed=False, color=my_color, thickness=2)
        else:
            for pt in pts_pix:
                cv2.circle(overlay, tuple(pt), radius=3, color=my_color, thickness=-1)
        if metrics is not None:
            label_lines = metric_labels.get(i, [f"{i}: {float(rewards[i]):.2f}"])
            label_anchor = pts_pix[-1]
            reward_labels.append({
                "label_lines": label_lines,
                "anchor": label_anchor,
                "color": my_color,
                "traj_idx": i,
            })

    # Sort by x coordinate
    reward_labels.sort(key=lambda item: item["anchor"][0])

    if reward_labels:
        trajectory_y_values = [item["anchor"][1] for item in reward_labels]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        pad = 4
        gap = 6
        bg_alpha = 0.55
        text_sizes = [
            cv2.getTextSize(line, font, font_scale, thickness)
            for item in reward_labels
            for line in item["label_lines"]
        ]
        text_w = max((size[0][0] for size in text_sizes), default=80)
        text_h = max((size[0][1] for size in text_sizes), default=15)
        baseline = max((size[1] for size in text_sizes), default=5)
        line_gap = 4
        max_lines = max((len(item["label_lines"]) for item in reward_labels), default=1)
        box_w = text_w + 2 * pad
        box_h = max_lines * text_h + (max_lines - 1) * line_gap + baseline + 2 * pad
        box_y_gap = 90

        top_y = int(min(trajectory_y_values)) if trajectory_y_values else 0
        box_top = int(np.clip(top_y - box_h - box_y_gap, 0, max(0, h - box_h - 1)))
        total_row_w = len(reward_labels) * box_w + max(0, len(reward_labels) - 1) * gap
        leftmost_anchor_x = min(int(item["anchor"][0]) for item in reward_labels)
        leftmost_anchor_x = min(leftmost_anchor_x, w//3)
        max_start_x = max(0, w - total_row_w - 1)
        current_x = int(np.clip(leftmost_anchor_x - box_w // 2, 0, max_start_x))

        for idx, item in enumerate(reward_labels):
            label_lines, anchor, color = item['label_lines'], item['anchor'], item['color']
            x = current_x
            best_box = (x, box_top, x + box_w, box_top + box_h)
            current_x = best_box[2] + gap

            anchor_xy = (int(anchor[0]), int(anchor[1]))
            label_center = ((best_box[0] + best_box[2]) // 2, (best_box[1] + best_box[3]) // 2)
            cv2.line(overlay, anchor_xy, label_center, color=color, thickness=1, lineType=cv2.LINE_AA)
            cv2.circle(overlay, anchor_xy, radius=2, color=color, thickness=-1)
            x1, y1, x2, y2 = best_box
            box_roi = overlay[y1:y2, x1:x2]
            black_fill = np.zeros_like(box_roi)
            cv2.addWeighted(black_fill, bg_alpha, box_roi, 1 - bg_alpha, 0, dst=box_roi)
            cv2.rectangle(overlay, (best_box[0], best_box[1]), (best_box[2], best_box[3]), color, thickness=1)
            for line_idx, line in enumerate(label_lines):
                line_y = box_top + pad + text_h + line_idx * (text_h + line_gap)
                cv2.putText(overlay, line, (x + pad, line_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return overlay


def load_calibration(json_path: str):
    """
    Builds:
      K (3x3), dist=None, T_cam_from_base (4x4)
    from tf.json with H_cam_bl: pitch(deg), x,y,z.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if data is None or "H_cam_bl" not in data:
        raise ValueError(f"Missing H_cam_bl in {json_path}")

    h = data["H_cam_bl"]
    roll = math.radians(float(h["roll"]))
    xt, yt, zt = float(h["x"]), float(h["y"]), float(h["z"])

    # Rotation about +y (camera pitched down is positive pitch if y up/right-handed)
    Ry = np.array([
        [0.0, math.sin(roll), math.cos(roll)],
        [-1.0, 0.0, 0.0],
        [0.0, -math.cos(roll), math.sin(roll)]
    ], dtype=np.float64)

    T_base_from_cam = np.eye(4, dtype=np.float64)
    T_base_from_cam[:3, :3] = Ry
    T_base_from_cam[:3, 3] = np.array([xt, yt, zt], dtype=np.float64)

    fx = data["Intrinsics"]["fx"]
    fy = data["Intrinsics"]["fy"]
    cx = data["Intrinsics"]["cx"]
    cy = data["Intrinsics"]["cy"]

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    dist = None  # explicitly no distortion
    return K, dist, T_base_from_cam

def get_action(diffusion_output, action_stats=ACTION_STATS):
    # diffusion_output: (B, 2*T+1, 1)
    # return: (B, T-1)
    device = diffusion_output.device
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = to_numpy(ndeltas)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(ndeltas, axis=1)
    return from_numpy(actions).to(device)