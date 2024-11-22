import os
import sys
import tqdm
import timm
import torch
import pathlib
import argparse
import pandas as pd
from PIL import Image
import huggingface_hub
from constants import *
import safetensors.torch
from typing import Optional

TORCH_DTYPE = torch.bfloat16
MIN_PROB = 0.5

def load_labels_hf(repo_id: str, revision: Optional[str]=None, token: Optional[str]=None) -> list[str]:
    try:
        csv_path = huggingface_hub.hf_hub_download(repo_id=repo_id, filename="selected_tags.csv", revision=revision, token=token)
        csv_path = pathlib.Path(csv_path).resolve()
    except huggingface_hub.utils.HfHubHTTPError as e:
        raise FileNotFoundError(f"\"selected_tags.csv\" failed to download from \"{repo_id}\"!") from e
    return pd.read_csv(csv_path, usecols=["name", "category"])["name"].tolist()

def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    # Convert to RGB/RGBA if not already(Deals with palette images etc).
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    # Convert RGBA to RGB with white background.
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image

def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    # Get the largest dimension so we can pad to a square.
    px = max(image.size)
    # Pad to square with white background.
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas

def get_image_tensor(path, transform, use_cuda=True):
    with Image.open(path) as img:
        return transform(pil_pad_square(pil_ensure_rgb(img))).unsqueeze(0)[:, [2, 1, 0]].to("cuda" if use_cuda else "cpu", TORCH_DTYPE)

class TaggingDataset(torch.utils.data.Dataset):

    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        return get_image_tensor(path, self.transform, False), path

def create_dataset(input_dir, transform, overwrite):
    paths = []
    for path in sorted(os.listdir(input_dir)):
        if path.endswith(".txt"):
            continue
        path = os.path.join(input_dir, path)
        if not os.path.isfile(path):
            continue
        if not overwrite and os.path.isfile(os.path.splitext(path)[0] + ".txt"):
            continue
        paths.append(path)
    return TaggingDataset(paths, transform)

@torch.no_grad
def tag(model, dataloader, model_labels, additional_tags):
    additional_tags = [tag.replace("_", " ").strip() for tag in additional_tags]
    for images, image_paths in tqdm.tqdm(dataloader, desc="Tagging Batch"):
        images = images.squeeze(1).to("cuda")
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        for image_path, prob in zip(image_paths, probs):
            rating_probs = prob[:4].detach().clone()
            prob[:4] = 0
            rating_tag_index = torch.max(rating_probs, -1).indices
            prob_indices = torch.nonzero(prob > MIN_PROB)
            tags = set(additional_tags)
            for i in prob_indices:
                tags.add(model_labels[i].replace("_", " "))
            tags = sorted(list(tags))
            tags_text = ", ".join([model_labels[rating_tag_index]] + tags)
            with open(os.path.splitext(image_path)[0] + ".txt", "w", encoding="utf8") as file:
                file.write(tags_text)

def parse_args():
    parser = argparse.ArgumentParser(description="Tag images using WD14 tagger.")
    parser.add_argument("-i", "--input-dir", default=IMAGE_DIR, help=f"The directory containing the images to tag, default to \"{IMAGE_DIR}\"")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="The amount of images to process at the same time, default to 32")
    parser.add_argument("-o", "--overwrite", action="store_true", help="If set, will overwrite the old tags")
    parser.add_argument("-a", "--additional-tags", nargs="+", default=[], help="Additional tags to add to the final tags")
    return parser.parse_args()

def main():
    args = parse_args()
    repo_id = "SmilingWolf/wd-eva02-large-tagger-v3"
    model = timm.create_model("hf-hub:" + repo_id)
    state_dict = safetensors.torch.load_file("model.safetensors")
    model.load_state_dict(state_dict)
    model.to("cuda", TORCH_DTYPE)
    model_labels = load_labels_hf(repo_id)
    transform = timm.data.create_transform(**timm.data.resolve_data_config(model.pretrained_cfg, model=model))
    dataset = create_dataset(args.input_dir, transform, args.overwrite)
    print("Tagging", len(dataset), "images...")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=os.cpu_count(), generator=torch.Generator().manual_seed(42))
    tag(model, dataloader, model_labels, args.additional_tags)
    print("Script finished.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user, exiting...")
        sys.exit(1)
