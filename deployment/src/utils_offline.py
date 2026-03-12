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
                 T_cam_from_base: Optional[np.ndarray] = None, color=(0, 0, 255)):
    if pts_cur.size == 0:
        return
    if cam_matrix is None or T_cam_from_base is None:
        return
    if img is None:
        return

    if len(pts_cur.shape) == 2:
        n_trajectories = 1
        pts_cur = np.expand_dims(pts_cur, 0)
    elif len(pts_cur.shape) == 3:
        n_trajectories = pts_cur.shape[0]
    else:
        print("error, unable to process pts_cur dimension", pts_cur.shape)

    # Points in base frame -> camera frame -> pixels
    R_cb = T_cam_from_base[:3, :3]
    t_cb = T_cam_from_base[:3, 3]
    rvec, _ = cv2.Rodrigues(R_cb)
    overlay = img.copy()
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
            return

        pts_pix = img_pts[keep].astype(int)
        if len(pts_pix) >= 2:
            cv2.polylines(overlay, [pts_pix], isClosed=False, color=color, thickness=2)
        else:
            for pt in pts_pix:
                cv2.circle(overlay, tuple(pt), radius=3, color=color, thickness=-1)

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