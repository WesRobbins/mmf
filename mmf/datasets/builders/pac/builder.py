import json
import logging
import math
import os
import zipfile

from collections import Counter

from mmf.common.registry import registry
from mmf.datasets.base_dataset_builder import BaseDatasetBuilder
# Let's assume for now that we have a dataset class called CLEVRDataset
from mmf.datasets.builders.pac.dataset import PACDataset
# from mmf.utils.general import download_file, get_mmf_root
from mmf.utils.general import get_mmf_root


logger = logging.getLogger(__name__)


@registry.register_builder("pac")
class PACBuilder(BaseDatasetBuilder):
    # DOWNLOAD_URL = ""https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"

    def __init__(self):
        # Init should call super().__init__ with the key for the dataset
        super().__init__("pac")

        # Assign the dataset class
        self.dataset_class = PACDataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/textvqa/defaults.yaml"

    def build(self, config, dataset):
        # download_folder = os.path.join(
        #     get_mmf_root(), config.data_dir, config.data_folder
        # )

        # file_name = self.DOWNLOAD_URL.split("/")[-1]
        # local_filename = os.path.join(download_folder, file_name)

        # extraction_folder = os.path.join(download_folder, ".".join(file_name.split(".")[:-1]))
        extraction_folder = '/content/drive/MyDrive/NEW_DATASET/'
        self.data_folder = extraction_folder

        # # Either if the zip file is already present or if there are some
        # # files inside the folder we don't continue download process
        # if os.path.exists(local_filename):
        #     return

        # if os.path.exists(extraction_folder) and \
        #     len(os.listdir(extraction_folder)) != 0:
        #     return

        # logger.info("Downloading the CLEVR dataset now")
        # download_file(self.DOWNLOAD_URL, output_dir=download_folder)

        # logger.info("Downloaded. Extracting now. This can take time.")
        # with zipfile.ZipFile(local_filename, "r") as zip_ref:
        #     zip_ref.extractall(download_folder)


    def load(self, config, dataset, *args, **kwargs):
        # Load the dataset using the CLEVRDataset class
        self.dataset = PACDataset(
            config, dataset, data_folder=self.data_folder
        )
        return self.dataset

    def update_registry_for_model(self, config):
        # Register both vocab (question and answer) sizes to registry for easy access to the
        # models. update_registry_for_model function if present is automatically called by
        # MMF
        registry.register(
            self.dataset_name + "_text_vocab_size",
            self.dataset.text_processor.get_vocab_size(),
        )
        registry.register(
            self.dataset_name + "_num_final_outputs",
            self.dataset.answer_processor.get_vocab_size(),
        )
        if hasattr(self.dataset, "answer_processor"):
            registry.register(
                self.dataset_name + "_num_final_outputs",
                self.dataset.answer_processor.get_vocab_size(),
            )
            registry.register(
                f"{self.dataset_name}_answer_processor", self.dataset.answer_processor
            )