import os
import copy
import glob
import cv2
import numpy as np
import onnxruntime
import argparse
from tqdm.auto import tqdm

# ==================== 配置参数 ====================
MODEL_PATH = "../D4/skyseg.onnx"
THRESHOLD = 32  # 天空分割阈值
INPUT_SIZE = [320, 320]  # 模型输入尺寸
# ==================================================

def run_skyseg(onnx_session, input_size, image):
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x. reshape(-1, 3, input_size[0], input_size[1]). astype("float32")
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session. get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})
    onnx_result = np.array(onnx_result). squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype("uint8")
    return onnx_result

def segment_sky(image_path, onnx_session, mask_filename, input_size=[320, 320], threshold=32):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    result_map = run_skyseg(onnx_session, input_size, image)
    result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))

        # 天空为白色(255)，背景为黑色(0)
    if SkyIsWhite:
        output_mask = np.zeros_like(result_map_original)
        output_mask[result_map_original > threshold] = 255
    else:
        # 天空为黑色(0)，背景为白色(255)
        output_mask = np.ones_like(result_map_original) * 255
        output_mask[result_map_original > threshold] = 0

    cv2.imwrite(mask_filename, output_mask)
    return output_mask

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='天空分割工具')
    parser.add_argument('-i', '--input', type=str, required=True, help='输入图片目录')
    parser.add_argument('-o', '--output', type=str, required=True, help='输出mask目录')
    parser.add_argument('-s', '--skywhite', action='store_true', help='天空是否为白色（默认False）')
    args = parser.parse_args()

    INPUT_DIR = args.input
    OUTPUT_DIR = args.output
    SkyIsWhite = args.skywhite

    # 检查并创建目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"模型: {MODEL_PATH}\n输入: {INPUT_DIR}\n输出: {OUTPUT_DIR}\n天空颜色: {'白色' if SkyIsWhite else '黑色'}\n")

    # 加载模型
    print("加载模型...")
    
    available_providers = onnxruntime.get_available_providers()
    # if "CUDAExecutionProvider" in available_providers:
    #     onnx_session = onnxruntime.InferenceSession(MODEL_PATH, providers=["CUDAExecutionProvider"])
    #     print("使用 GPU (CUDAExecutionProvider)")
    # else:
    onnx_session = onnxruntime.InferenceSession(MODEL_PATH)
    print("使用 CPU")
    print("✓ 模型加载成功\n")

    # 获取所有图片
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    image_files = sorted(image_files)
    print(f"找到 {len(image_files)} 张图片\n")

    # 批量处理
    success_count = 0
    total_sky_percentage = 0

    for image_path in tqdm(image_files, desc="处理进度"):
        try:
            filename = os.path.basename(image_path)
            output_path = os.path.join(OUTPUT_DIR, filename)
            mask = segment_sky(image_path, onnx_session, output_path, INPUT_SIZE, THRESHOLD)

            # 根据 SkyIsWhite 统计天空像素
            if SkyIsWhite:
                sky_pixels = np.sum(mask == 255)
            else:
                sky_pixels = np.sum(mask == 0)

            total_pixels = mask.shape[0] * mask.shape[1]
            total_sky_percentage += (sky_pixels / total_pixels) * 100
            success_count += 1
        except Exception as e:
            print(f"\n处理失败: {filename} - {e}")

    print(f"\n完成！成功: {success_count} 张")
    if success_count > 0:
        print(f"平均天空占比: {total_sky_percentage/success_count:.2f}%")
    print(f"输出: {OUTPUT_DIR}")