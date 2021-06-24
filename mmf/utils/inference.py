# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import requests
import torch
from mmf.common.report import Report
from mmf.common.sample import Sample, SampleList
from mmf.utils.build import build_encoder, build_model, build_processors
from mmf.utils.checkpoint import load_pretrained_model
from mmf.utils.general import get_current_device
from omegaconf import OmegaConf
from PIL import Image

import omegaconf
from mmf.common.registry import registry
from mmf.datasets.multi_datamodule import MultiDataModule
from mmf.datasets.builders.coco.builder import COCOBuilder
from mmf.datasets.builders.textvqa.builder import TextVQABuilder


class Inference:
    def __init__(self, checkpoint_path: str = None):
        self.checkpoint = checkpoint_path
        assert self.checkpoint is not None
        self.dataset_loader = MultiDataModule(registry.get("config"))
        self.processor, self.feature_extractor, self.model = self._build_model()

    def _build_model(self):
        self.model_items = load_pretrained_model(self.checkpoint)
        self.config = OmegaConf.create(self.model_items["full_config"])
        dataset_name = list(self.config.dataset_config.keys())[0]
        print("PROCESSORS")
        print(self.config.dataset_config[dataset_name].processors)
        processor = build_processors(
            self.config.dataset_config[dataset_name].processors
        )
        self.img_feature_encodings_config = self.model_items["config"].image_feature_encodings
        if type(self.img_feature_encodings_config) == omegaconf.listconfig.ListConfig:
            self.img_feature_encodings_config = self.img_feature_encodings_config[0]
        feature_extractor = build_encoder(
            #self.model_items["config"].image_feature_encodings
            self.img_feature_encodings_config
        )
        ckpt = self.model_items["checkpoint"]
        model = build_model(self.model_items["config"])
        model.load_state_dict(ckpt, strict=False)

        return processor, feature_extractor, model

    def forward(self, image_path: str, text: dict, image_format: str = "path"):
        text_output = self.processor["text_processor"](text)
        if image_format == "path":
            img = np.array(Image.open(image_path))
        elif image_format == "url":
            img = np.array(Image.open(requests.get(image_path, stream=True).raw))
        img = torch.as_tensor(img)

        # if self.model_items["config"].image_feature_encodings.type == "frcnn":
        if self.img_feature_encodings_config.type == "frcnn" or "finetune_faster_rcnn_fpn_fc7":
            # max_detect = self.model_items[
            #     "config"
            # ].image_feature_encodings.params.max_detections
            max_detect = self.img_feature_encodings_config.params.max_detections

            pre_config = omegaconf.OmegaConf.load('/content/drive/MyDrive/data/mmf/models/frcnn/config2.yaml')
            from tools.scripts.features.frcnn.processing_image import Preprocess
            image_processor = Preprocess(pre_config)
            # image_preprocessed, sizes, scales_yx = self.processor["image_processor"](
            #     img
            # )
            image_preprocessed, sizes, scales_yx = image_processor(img)

            from mmf.modules.encoders import FRCNNImageEncoder
            frcnn = FRCNNImageEncoder(self.img_feature_encodings_config)
            frcnn.eval()

            print(self.feature_extractor)
            image_output = frcnn(
                image_preprocessed,
                sizes=sizes,
                scales_yx=scales_yx,
                padding=None,
                max_detections=max_detect,
                return_tensors="pt",
            )
            print(image_output["roi_features"].size)
            image_output = image_output["roi_features"]
        else:
            print(self.processor)
            image_preprocessed = self.processor["image_processor"](img)
            image_output = self.feature_extractor(image_preprocessed)

        sample = Sample(text_output)
        sample.image_feature_0 = image_output
        sample_list = SampleList([sample])
        sample_list = sample_list.to(get_current_device())
        self.model = self.model.to(get_current_device())
        output = self.model(sample_list)
        sample_list.id = [sample_list.input_ids[0][0]]
        report = Report(sample_list, output)
        answers = self.processor["output_processor"](report)
        answer = self.processor["answer_processor"].idx2word(answers[0]["answer"])

        return answer
