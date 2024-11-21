import os
import io
import sys
import json
import tqdm
import hashlib
import asyncio
import aiohttp
import argparse
import concurrent
from PIL import Image

MAX_TASKS = 50
MAX_RETRY = 3
TIMEOUT = 10

IMAGE_EXT = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif",
    ".webp", ".heic", ".heif", ".avif", ".jxl",
}

def val_and_save(data, output_dir, url, ext):
    try:
        with io.BytesIO(data) as filelike:
            with Image.open(filelike) as img:
                hasher = hashlib.md5()
                hasher.update(data)
                if not ext:
                    img.save(os.path.join(output_dir, hasher.hexdigest() + ".png"))
                else:
                    img.verify()
                    with open(os.path.join(output_dir, hasher.hexdigest() + ext), "wb") as file:
                        file.write(data)
    except Exception as e:
        print(f"Error validating image \"{url}\": {e}")

async def download(url, session, thread_pool, output_dir):
    error = None
    for i in range(1, MAX_RETRY + 2):
        try:
            ext = os.path.splitext(url)[1].lower()
            if ext and ext not in IMAGE_EXT:
                print(f"Image \"{url}\" is not an image, skipped.")
                return
            async with session.get(url) as response:
                data = await response.read()
            await asyncio.wrap_future(thread_pool.submit(val_and_save, data, output_dir, url, ext))
            return
        except Exception as e:
            error = e
            if i > MAX_RETRY:
                break
            # print(f"A {e.__class__.__name__} occurred with image \"{url}\": {e}\nPausing for 0.1 second before retrying attempt {i}/{MAX_RETRY}...")
            await asyncio.sleep(0.1)
    if error is not None:
        print(f"All retry attempts failed, image \"{url}\" skipped. Final error {error.__class__.__name__}: {error}")

def parse_args():
    parser = argparse.ArgumentParser(description="Mass download images using AsyncIO.")
    parser.add_argument("-i", "--input", default="targets.json", help="The JSON file containing a list of URLs to download, default to \"targets.json\"")
    parser.add_argument("-o", "--output-dir", default="images", help="The directory to download the images into, default to \"images\"")
    return parser.parse_args()

async def main():
    args = parse_args()
    with open(args.input, "r", encoding="utf8") as file:
        targets = json.load(file)
    assert isinstance(targets, list), f"The file \"{args.input}\" must be a JSON list!"
    os.makedirs(args.output_dir, exist_ok=True)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as session:
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as thread_pool, tqdm.tqdm(total=len(targets), desc="Downloading") as pbar:
            tasks = []
            for url in targets:
                while len(tasks) >= MAX_TASKS:
                    await asyncio.sleep(0.1)
                    for i in range(len(tasks) - 1, -1, -1):
                        task = tasks[i]
                        if task.done():
                            await task
                            del tasks[i]
                            pbar.update(1)
                tasks.append(asyncio.create_task(download(url, session, thread_pool, args.output_dir)))
            while tasks:
                await asyncio.sleep(0.1)
                for i in range(len(tasks) - 1, -1, -1):
                    task = tasks[i]
                    if task.done():
                        await task
                        del tasks[i]
                        pbar.update(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScript interrupted by user, exiting...")
        sys.exit(1)
