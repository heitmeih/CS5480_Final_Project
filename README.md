# CS5480 Final Project: Drone-based Human and Tent Detection

## Environment Setup

### `uv` (Recommended)

We are using `uv` to manage our environment. Click here for installation instructions: <https://docs.astral.sh/uv/getting-started/installation/>. You can then sync your environment with the following command:

```sh
uv sync
```

To run files, use `uv run`, such as

```sh
uv run main.py
```

### Alternatives (Untested)

Alternatively, you may try using a method of your choice and install dependencies via the requirements file:

```sh
pip install -r requirements.txt
```

If you use Conda, you can create a new environment and download `uv` with `pip install uv` and setup your environment using the recommended method above.

## Training

To run the training scripts, do the following:

```sh
uv run train_resnet.py
# or
uv run train_mobilenet.py
```

> If you are not using `uv`, run the scripts as you normally would (such as `python train_resnet.py`).

These will automatically download the necessary datasets. The automatic download script downloads the data from Google Drive; if this fails for any reason, here are instructions to download the data:

1. Download the following datasets from Roboflow as `zip` files in the COCO annotation format:
   - SARD (images with humans): <https://universe.roboflow.com/datasets-pdabr/sard-8xjhy/dataset/9>
   - tent (images with tents): <https://universe.roboflow.com/aa-44pfv/tent-budfp/dataset/1>
2. extract each dataset into their corresponding folders (paths are relative to repository root):
   - SARD: extract into `./data/sard`
   - tent: extract into `./data/tent`
   - **Note:** the resulting structure should be `./data/{dataset}/{split}` after this is complete, where `{split}` is `test`, `train`, or `valid`.

## Inference

After training the model, to run inference on single images, do the following:

```sh
uv run infer_one_resnet.py path/to/test/image.jpg
# or
uv run infer_one_mobilenet.py path/to/test/image.jpg
```

These scripts will show the detect objects with labeled bounding boxes.
