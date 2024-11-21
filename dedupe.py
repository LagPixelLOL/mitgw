import os
import sys
import time
import tqdm
import shutil
import skimage
import argparse
import threading
import numpy as np
import scipy as sp
from PIL import Image
from constants import *
import concurrent.futures

def convert_to_rgb(image):
    # `image.convert("RGB")` would only work for .jpg images, as it creates a wrong
    # background for transparent images. The call to `alpha_composite` handles this case.
    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite

def rmse(org_img: np.ndarray, pred_img: np.ndarray, max_p: int=4095) -> float:
    org_img = org_img.astype(np.float32)
    if org_img.ndim == 2:
        org_img = np.expand_dims(org_img, axis=-1)
    rmse_bands = []
    diff = org_img - pred_img
    mse_bands = np.mean(np.square(diff / max_p), axis=(0, 1))
    rmse_bands = np.sqrt(mse_bands)
    return np.mean(rmse_bands)

def ssim(org_img: np.ndarray, pred_img: np.ndarray, max_p: int=4095) -> float:
    return skimage.metrics.structural_similarity(org_img, pred_img, data_range=max_p, channel_axis=2)

def get_img(path):
    with Image.open(path) as img:
        return np.array(convert_to_rgb(img).resize((512, 512), Image.LANCZOS))

def img_similarity(img_a, img_a_path, img_b_path, lock, groups, output_dir):
    img_b = get_img(img_b_path)
    if rmse(img_a, img_b) < 0.01 or ssim(img_a, img_b) > 0.95:
        img_a_basename = os.path.basename(img_a_path)
        img_b_basename = os.path.basename(img_b_path)
        with lock:
            for i in range(len(groups)):
                group = groups[i]
                if img_a_basename in group or img_b_basename in group:
                    break
            else:
                i = len(groups)
                group = set()
                groups.append(group)
            group_dir = os.path.join(output_dir, str(i))
            os.makedirs(group_dir, exist_ok=True)
            shutil.copy(img_a_path, group_dir)
            shutil.copy(img_b_path, group_dir)
            group.add(img_a_basename)
            group.add(img_b_basename)

def parse_args():
    parser = argparse.ArgumentParser(description="Group images together to inspect.")
    parser.add_argument("-i", "--input-dir", default=IMAGE_DIR, help=f"The directory containing the input images, default to \"{IMAGE_DIR}\"")
    parser.add_argument("-o", "--output-dir", default=GROUP_DIR, help=f"The directory to save the grouped images, default to \"{GROUP_DIR}\"")
    parser.add_argument("-d", "--delete-mode", action="store_true", help="If set, will get all the filenames remaining in the output directory and delete them from the input directory")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.delete_mode:
        i = 0
        for group_basename in os.listdir(args.output_dir):
            group_dir = os.path.join(args.output_dir, group_basename)
            for img_basename in os.listdir(group_dir):
                os.remove(os.path.join(args.input_dir, img_basename))
                os.remove(os.path.join(group_dir, img_basename))
                i += 1
            os.rmdir(group_dir)
        os.rmdir(args.output_dir)
        print("Deleted", i, "images, script finished.")
        return
    paths = [os.path.join(args.input_dir, path) for path in os.listdir(args.input_dir)]
    path_count = len(paths)
    os.makedirs(args.output_dir, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as thread_pool, tqdm.tqdm(total=sp.special.comb(path_count, 2, exact=True), desc="Grouping") as pbar:
        lock = threading.Lock()
        groups = []
        for i in range(path_count):
            img_a_path = paths[i]
            futures = []
            img_a = get_img(img_a_path)
            for j in range(i + 1, path_count):
                futures.append(thread_pool.submit(img_similarity, img_a, img_a_path, paths[j], lock, groups, args.output_dir))
            while futures:
                time.sleep(0.1)
                for k in range(len(futures) - 1, -1, -1):
                    if futures[k].done():
                        del futures[k]
                        pbar.update(1)
    print("Script finished.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user, exiting...")
        sys.exit(1)
