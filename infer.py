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
from tqdm import tqdm
DEFAULT_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
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
        match = re.search(r"frame_(\d+)", basename)
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


def read_frames_list_from_dir(frame_dir):
    """从目录中读取帧序列，支持 frame_{frameid}.jpg 格式"""
    frame_files = glob.glob(os.path.join(frame_dir, "frame_*.jpg"))
    frame_files += glob.glob(os.path.join(frame_dir, "frame_*.png"))

    if not frame_files:
        raise ValueError(f"在 {frame_dir} 中没有找到 frame_*.jpg 或 frame_*.png 文件")

    # 提取帧ID并排序
    def extract_frame_id(filepath):
        basename = os.path.basename(filepath)
        match = re.search(r"frame_(\d+)", basename)
        if match:
            return int(match.group(1))
        return 0

    frame_files.sort(key=extract_frame_id)

    return frame_files


def LoadFrameFromFileList(frame_files):
    frames = []
    for frame_file in frame_files:
        frame = np.array(Image.open(frame_file))
        frames.append(frame)

    print(f"从文件列表中读取了 {len(frames)} 帧")
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
    parser.add_argument(
        "-c",
        "--chunksize",
        type=int,
        default=30,
        help="number of frames to be processed in a chunk",
    )
    parser.add_argument("-s", "--save_dir", type=str, default="./saved_datas", help="directory to save output video")
    args = parser.parse_args()

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
    # load the input video frame by frame
    if args.frame_dir is None:
        exit("请提供 --frame_dir 参数，指向包含 frame_*.jpg/png 文件的目录")

    frame_files = read_frames_list_from_dir(args.frame_dir)
    videochunks = (len(frame_files) + args.chunksize - 1) // args.chunksize
    for chunkidx in tqdm(range(videochunks), desc="Processing chunks"):
        startindex = chunkidx * args.chunksize
        endindex = min((chunkidx + 1) * args.chunksize, len(frame_files))
        chunk_frame_files = frame_files[startindex:endindex]
        video = LoadFrameFromFileList(chunk_frame_files)
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
        video = video.to(DEFAULT_DEVICE)
        firstFrameName = os.path.basename(chunk_frame_files[0])
        base_name,_ = os.path.splitext(firstFrameName)
        sargs = base_name.split("_")
        num = int(sargs[-1])
        grid_query_frame = args.grid_query_frame if startindex + args.grid_query_frame < endindex else 0
        # 1->000001
        base_name = f'{sargs[0]}_{(num + grid_query_frame):06d}'
        firstMaskPath = os.path.join(args.mask_path, base_name + '.png')
        print("Using mask:", firstMaskPath)
        segm_mask = np.array(Image.open(firstMaskPath))
        segm_mask = torch.from_numpy(segm_mask)[None, None]
        pred_tracks, pred_visibility = model(
            video,
            grid_size=args.grid_size,
            grid_query_frame=grid_query_frame,
            backward_tracking=args.backward_tracking,
            segm_mask=segm_mask,
        )
        # 这里顺便输出一下用时
        save_path = os.path.join(args.save_dir, f"{startindex}_{endindex}")
        os.makedirs(save_path, exist_ok=True)
        np.savez_compressed(os.path.join(save_path, "track.npz"), tracks=pred_tracks.cpu().numpy(), visibility=pred_visibility.cpu().numpy())
        vis = Visualizer(save_dir=f"./saved_videos/{startindex}_{endindex}", pad_value=120, linewidth=3)
        vis.visualize(
            video,
            pred_tracks,
            pred_visibility,
            query_frame=0 if args.backward_tracking else args.grid_query_frame,
        )
    print("computed")
    
    # # save a video with predicted tracks
    # vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
    # vis.visualize(
    #     video,
    #     pred_tracks,
    #     pred_visibility,
    #     query_frame=0 if args.backward_tracking else args.grid_query_frame,
    # )
