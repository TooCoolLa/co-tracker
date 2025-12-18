# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import numpy as np
import glob
import re

from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# if DEFAULT_DEVICE == "mps":
#     os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def read_frames_from_dir(frame_dir):
    """从目录中读取帧序列，支持 frame_{frameid}.jpg 格式"""
    frame_files = glob.glob(os.path.join(frame_dir, "frame_*.jpg"))
    frame_files += glob.glob(os.path.join(frame_dir, "frame_*.png"))
    
    if not frame_files:
        raise ValueError(f"在 {frame_dir} 中没有找到 frame_*.jpg 或 frame_*.png 文件")
    
    # 提取帧ID并排序
    def extract_frame_id(filepath):
        basename = os.path.basename(filepath)
        match = re.search(r'frame_(\d+)', basename)
        if match:
            return int(match.group(1))
        return 0
    
    frame_files.sort(key=extract_frame_id)
    
    frames = []
    for frame_file in frame_files:
        frame = np.array(Image.open(frame_file))
        frames.append(frame)
    
    print(f"从 {frame_dir} 读取了 {len(frames)} 帧")
    return np.stack(frames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="./assets/apple.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--frame_dir",
        default=None,
        help="path to a directory containing frame_*.jpg/png files (alternative to video_path)",
    )
    parser.add_argument(
        "--mask_path",
        default="./assets/apple_mask.png",
        help="path to a segmentation mask",
    )
    parser.add_argument(
        "--checkpoint",
        # default="./checkpoints/cotracker.pth",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Compute tracks in both directions, not only forward",
    )
    parser.add_argument(
        "--use_v2_model",
        action="store_true",
        help="Pass it if you wish to use CoTracker2, CoTracker++ is the default now",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Pass it if you would like to use the offline model, in case of online don't pass it",
    )

    args = parser.parse_args()

    # load the input video frame by frame
    if args.frame_dir is not None:
        video = read_frames_from_dir(args.frame_dir)
        seq_name = os.path.basename(args.frame_dir.rstrip('/'))
    else:
        video = read_video_from_path(args.video_path)
        seq_name = args.video_path.split("/")[-1]
    
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    segm_mask = np.array(Image.open(os.path.join(args.mask_path)))
    segm_mask = torch.from_numpy(segm_mask)[None, None]

    if args.checkpoint is not None:
        if args.use_v2_model:
            model = CoTrackerPredictor(checkpoint=args.checkpoint, v2=args.use_v2_model)
        else:
            if args.offline:
                window_len = 60
            else:
                window_len = 16
            model = CoTrackerPredictor(
                checkpoint=args.checkpoint,
                v2=args.use_v2_model,
                offline=args.offline,
                window_len=window_len,
            )
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")

    model = model.to(DEFAULT_DEVICE)
    video = video.to(DEFAULT_DEVICE)

    pred_tracks, pred_visibility = model(
        video,
        grid_size=args.grid_size,
        grid_query_frame=args.grid_query_frame,
        backward_tracking=args.backward_tracking,
        segm_mask=segm_mask
    )
    # 这里顺便输出一下用时

    print("computed")

    # save a video with predicted tracks
    vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
    vis.visualize(
        video,
        pred_tracks,
        pred_visibility,
        query_frame=0 if args.backward_tracking else args.grid_query_frame,
    )
