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
from concurrent.futures import ThreadPoolExecutor

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
    def load_frame(frame_file):
        return np.array(Image.open(frame_file))

    # Using ThreadPoolExecutor inside here as well for parallel file reading
    with ThreadPoolExecutor() as executor:
        frames = list(executor.map(load_frame, frame_files))

    # print(f"从文件列表中读取了 {len(frames)} 帧")
    return np.stack(frames)


def save_and_visualize(save_dir, video_path, startindex, endindex, tracks, visibility, video_tensor, query_frame):
    """
    Background task to save results and visualize.
    Everything here should be on CPU.
    """
    save_path = os.path.join(save_dir, f"{startindex}_{endindex}")
    os.makedirs(save_path, exist_ok=True)
    
    # Save npz
    np.savez_compressed(
        os.path.join(save_path, "track.npz"), 
        tracks=tracks, 
        visibility=visibility
    )
    
    # Visualize
    vis = Visualizer(save_dir=f"./saved_videos/{startindex}_{endindex}", pad_value=120, linewidth=3)
    vis.visualize(
        video_tensor, # Ensure this is on CPU
        torch.from_numpy(tracks),      # Visualizer expects tensor or numpy? Check existing usage. 
                                     # Existing usage: pred_tracks (tensor on GPU)
                                     # But here we passed numpy. Let's check Visualizer code or just pass tensors back.
                                     # To be safe and avoid passing GPU tensors to thread, we pass CPU tensors.
                                     # Visualizer usually handles CPU tensors fine.
        torch.from_numpy(visibility),
        query_frame=query_frame,
    )
    # print(f"Finished saving and visualizing for chunk {startindex}_{endindex}")


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
    
    if args.frame_dir is None:
        exit("请提供 --frame_dir 参数，指向包含 frame_*.jpg/png 文件的目录")

    frame_files = read_frames_list_from_dir(args.frame_dir)
    videochunks = (len(frame_files) + args.chunksize - 1) // args.chunksize
    
    # Create a thread pool for background saving/visualization
    # Adjust max_workers as needed. Too high might cause OOM or IO contention.
    save_executor = ThreadPoolExecutor(max_workers=16) 
    futures = []

    for chunkidx in tqdm(range(videochunks), desc="Processing chunks"):
        startindex = chunkidx * args.chunksize
        endindex = min((chunkidx + 1) * args.chunksize, len(frame_files))
        chunk_frame_files = frame_files[startindex:endindex]
        save_path = os.path.join(args.save_dir, f"{startindex}_{endindex}")
        if os.path.exists(save_path):
            print(f"Skipping already processed chunk {startindex}_{endindex}")
            continue
        # Parallel Image Loading (IO Bound)
        video_np = LoadFrameFromFileList(chunk_frame_files)
        video = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].float()
        video = video.to(DEFAULT_DEVICE)
        
        firstFrameName = os.path.basename(chunk_frame_files[0])
        base_name,_ = os.path.splitext(firstFrameName)
        sargs = base_name.split("_")
        num = int(sargs[-1])
        grid_query_frame = args.grid_query_frame if startindex + args.grid_query_frame < endindex else 0
        
        # Mask Loading
        base_name = f'{sargs[0]}_{(num + grid_query_frame):06d}'
        firstMaskPath = os.path.join(args.mask_path, base_name + '.png')
        # print("Using mask:", firstMaskPath)
        segm_mask = np.array(Image.open(firstMaskPath))
        segm_mask = torch.from_numpy(segm_mask)[None, None]
        
        # Inference (GPU Bound)
        pred_tracks, pred_visibility = model(
            video,
            grid_size=args.grid_size,
            grid_query_frame=grid_query_frame,
            backward_tracking=args.backward_tracking,
            segm_mask=segm_mask,
        )
        
        # Move results to CPU for async processing
        tracks_cpu = pred_tracks.cpu().numpy()
        # n,pointcount
        visibility_cpu = pred_visibility.cpu().numpy()
        video_cpu = video.cpu() # Tensor
        visdata = visibility_cpu[0]
        trackdata = tracks_cpu[0]
        for i in range(visdata.shape[0]):
            # 获取当前帧的文件名（不含扩展名），使用 basename 避免路径拼接错误
            frame_path = chunk_frame_files[i]
            base_name = os.path.splitext(os.path.basename(frame_path))[0]
            mask_path = os.path.join(args.mask_path, base_name + '.png')
            
            if not os.path.exists(mask_path):
                continue
            
            mask = np.array(Image.open(mask_path))
            
            # 获取当前帧所有点的坐标 (N, 2)
            coords = trackdata[i]
            
            # 四舍五入取整并转换为整数索引
            x = np.rint(coords[:, 0]).astype(np.int32)
            y = np.rint(coords[:, 1]).astype(np.int32)
            
            h, w = mask.shape[:2]
            
            # 筛选出在图像范围内的有效点索引
            valid_idx = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            
            # 检查Mask值，黑色(0)为不可见
            # 支持灰度图(ndim=2)和RGB图(ndim=3)
            if mask.ndim == 2:
                is_black = (mask[y[valid_idx], x[valid_idx]] == 0)
            else:
                is_black = np.all(mask[y[valid_idx], x[valid_idx]] == 0, axis=-1)
            
            # 更新visibility
            # 创建一个需要置为False的点的布尔掩码 (N,)
            to_hide = np.zeros(visdata.shape[1], dtype=bool)
            to_hide[valid_idx] = is_black
            
            # 将对应点的visibility设置为False
            visdata[i, to_hide] = False

        query_frame_val = 0 if args.backward_tracking else grid_query_frame
        
        # Submit task to background thread
        future = save_executor.submit(
            save_and_visualize,
            args.save_dir,
            f"./saved_videos/{startindex}_{endindex}", # Note: logic inside uses this as base
            startindex,
            endindex,
            tracks_cpu,
            visibility_cpu,
            video_cpu,
            query_frame_val
        )
        futures.append(future)
        torch.cuda.empty_cache()  # Clear GPU memory after each chunk to avoid OOM
    print("Inference loop finished. Waiting for background tasks...")
    # Wait for all background tasks to complete
    for f in futures:
        f.result() # This will also raise any exceptions that occurred in threads
        
    save_executor.shutdown()
    print("All computed and saved.")
