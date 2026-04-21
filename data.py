from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import gdown
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms
from torchvision.datasets import CocoDetection

data_dir: Path = Path(__file__).parent / "data"

coco_categories = [{"id": 0, "name": "person"}, {"id": 1, "name": "tent"}]

coco_name_id_map = {cat["name"]: cat["id"] for cat in coco_categories}


class RemappedDataset(Dataset):

    def __init__(self, dataset: Dataset, remapping: dict[int, int]):
        self.dataset = dataset
        self.remapping = remapping

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, annotations = self.dataset[index]

        for ann in annotations:
            ann["category_id"] = self.remapping.get(
                ann["category_id"], ann["category_id"]
            )

        return image, annotations


@dataclass
class DataSource:

    name: str
    manual_download_alt: str
    drive_download_uri: str | None = None
    file_ext: str = "zip"
    id_remapping: dict[int, int] | None = None

    @property
    def filename(self):
        return f"{self.name}.{self.file_ext}"


coco_data_sources: list[DataSource] = [
    DataSource(
        "tent",
        "https://universe.roboflow.com/aa-44pfv/tent-budfp/dataset/1",
        "https://drive.google.com/uc?id=1iqT4J7VldqKcj8MYMc5IPioYMc-4DgzN",
        id_remapping={1: coco_name_id_map["tent"]},
    ),
    DataSource(
        "sard",
        "https://universe.roboflow.com/datasets-pdabr/sard-8xjhy/dataset/9",
        "https://drive.google.com/uc?id=1-eKBVrvFjPfFCTV709xSNj_LU9iG3Slx",
        id_remapping={1: coco_name_id_map["person"]},
    ),
]

all_data_sources: list[DataSource] = [*coco_data_sources]


def _log_manual_download_process(source: DataSource, directory: str):
    print(
        f"Cannot download {source.name} dataset automatically. Download it manually here, using COCO format if available (such as on Roboflow): {source.manual_download_alt}",
        f"Unzip the data into folder {directory}.",
    )


def download_sources(
    sources: Iterable[DataSource],
    download_dir: Path | str,
    extracted_dir: Path | str | None = None,
):
    """Attempts to download `DataSource`s from convinient Google Drive location. Will
    print instructions for manually downloads if the source is unavailable for downloading in
    this way.

    Args:
        sources: The `DataSource`s to download from.
        download_dir: The folder in which to download raw source files. These files are removed after extraction.
        extracted_dir: The folder in which to extract the raw source. If not provided, will be same as download_dir
    """
    if not download_dir.exists():
        download_dir.mkdir(parents=True)

    if extracted_dir is None:
        extracted_dir = download_dir

    for source in sources:
        extracted_source_dir = Path(extracted_dir) / source.name

        if not extracted_source_dir.exists():
            extracted_source_dir.mkdir(parents=True)

        extracted_source_dir = str(extracted_source_dir)

        if not source.drive_download_uri:
            _log_manual_download_process(source, extracted_source_dir)
            continue

        outfile = Path(download_dir) / source.filename

        try:
            # download file
            print("Downloading", source.name)
            gdown.download(source.drive_download_uri, str(outfile))

            # unzip file
            print("Extracting", source.name)
            gdown.extractall(str(outfile), extracted_source_dir)

            # delete zip archive
            print("Removing Download for", source.name)
            outfile.unlink()
        except Exception:
            print("Error: could not download dataset", source.name)
            _log_manual_download_process(source, extracted_source_dir)


def load_datasets(
    sources: Iterable[DataSource] | None = None,
    data_folder: str | Path | None = None,
    json_name="_annotations.coco.json",
):
    if sources is None:
        sources = all_data_sources

    if data_folder is None:
        data_folder = data_dir

    subsets = []

    for source in sources:
        source_dir = Path(data_folder) / source.name
        for subfolder in "train", "test", "valid":
            dataset = CocoDetection(
                source_dir / subfolder,
                str(source_dir / subfolder / json_name),
                transform=transforms.ToTensor(),
            )

            if source.id_remapping:
                dataset = RemappedDataset(dataset, source.id_remapping)

            subsets.append(dataset)

    return ConcatDataset(subsets)


if __name__ == "__main__":
    import sys

    if sys.argv[1].lower() == "download":
        # TODO: make this configurable, such as downloading specific sources
        download_sources(all_data_sources, data_dir)
    else:
        print("Did not recognize command.")
