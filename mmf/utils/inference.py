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
from mmf.datasets.builders.textcaps.builder import TextCapsBuilder


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
        model.eval()
        model.load_state_dict(ckpt, strict=False)

        return processor, feature_extractor, model

    def forward(self, image_path: str, text: dict, image_format: str = "path"):
        text_output = self.processor["text_processor"](text)
        if image_format == "path":
            img = np.array(Image.open(image_path))
        elif image_format == "url":
            img = np.array(Image.open(requests.get(image_path, stream=True).raw))
        img = torch.as_tensor(img)

        need = False
        if need:
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

                frcnn_ckpt = torch.load('/content/drive/MyDrive/data/mmf/models/frcnn/pytorch_model.bin')
                frcnn_ckpt2 = {}
                for i in frcnn_ckpt.keys():
                    key = "frcnn."+i
                    frcnn_ckpt2[key] = frcnn_ckpt[i]
                print(type(frcnn_ckpt2))
                #print(ckpt.keys())

                from mmf.modules.encoders import FRCNNImageEncoder
                frcnn = FRCNNImageEncoder(pre_config)#(self.img_feature_encodings_config)
                frcnn.load_state_dict(frcnn_ckpt2)
                frcnn.eval()

                print(self.feature_extractor)
                output = frcnn(
                    image_preprocessed,
                    sizes=sizes,
                    scales_yx=scales_yx,
                    padding=None,
                    max_detections=max_detect,
                    return_tensors="pt",
                )
                print(type(output))
                print(output.keys())
                print(output["roi_features"].size())
                image_output = output["roi_features"].squeeze(0)
                # obj_boxes = output["normalized_boxes"].squeeze(0)
                obj_boxes = output["boxes"].squeeze(0)
                print(output["obj_ids"])
                print(output["obj_probs"])
                print(output["attr_ids"])
                print(output["attr_probs"])
                
            else:
                print(self.processor)
                image_preprocessed = self.processor["image_processor"](img)
                image_output = self.feature_extractor(image_preprocessed)

        # if features already saved in as npy
        else:
            img = np.load('/content/drive/MyDrive/data/mmf/datasets/new/features/image.npy', allow_pickle=True)
            img_info = np.load('/content/drive/MyDrive/data/mmf/datasets/new/features/image_info.npy', allow_pickle=True)

            image_output = torch.from_numpy(img)
            obj_boxes = img_info.item()['bbox']
            sums = obj_boxes.sum(axis=1)
            obj_boxes = obj_boxes/ sums[:, np.newaxis]
            obj_boxes = torch.from_numpy(obj_boxes) 


        sample = Sample(text_output)
        sample.text_len = torch.tensor(len(sample.text))
        sample.text = sample.input_ids
        sample.image_feature_0 = image_output#[:2]
        sample.obj_bbox_coordinates = obj_boxes#[:2]

        sample.train_prev_inds = torch.zeros(12, dtype=torch.int64)

        
        sample.image_info_0 = Sample()
        sample.image_info_0.max_features = torch.tensor(image_output.size(1))

        NUM_OCR = 0
        MAX_OCR = 0

        from  mmf.datasets.processors.processors import FastTextProcessor, SimpleWordProcessor
        cfg = registry.get("config")
        simple = SimpleWordProcessor()
        tokens = simple({"text": "this is Joe Biden's"})
        print(tokens)

        fasttext = FastTextProcessor(cfg.dataset_config.textvqa.processors.context_processor.params)
        print(fasttext)
        context = fasttext({"tokens": tokens})
        print(context)
        import sys 
        sys.exit()

        sample.image_feature_1 = torch.zeros(NUM_OCR,2048)
        sample.image_info_1 = Sample()
        sample.image_info_1.max_features = torch.tensor(MAX_OCR)#(image_output.size(1))

        sample.context = torch.zeros(NUM_OCR,300)
        sample.context_tokens = torch.zeros(0)
        sample.context_feature_0 = torch.zeros(NUM_OCR,300)
        sample.context_info_0 = Sample()
        sample.context_info_0.max_features = torch.tensor(50)

        sample.context_feature_1 = torch.zeros(NUM_OCR, 604)
        sample.order_vectors = torch.zeros(NUM_OCR,50)       # this may be deprecated
        sample.ocr_bbox_coordinates = torch.zeros(NUM_OCR,4)

        sample_list = SampleList([sample])
        sample_list = sample_list.to(get_current_device())
        self.model = self.model.to(get_current_device())
        output = self.model(sample_list)
        sample_list.id = [sample_list.input_ids[0][0]]
        report = Report(sample_list, output)
        print("SCORES")
        print(report.scores.size())
        print(report.scores)
        print(report.id)
        # print(report.id.size())
        print(len(report.id))

        # from mmf.datasets.processors.prediction_processors import ArgMaxPredictionProcessor
        # self.processor["output_processor"] = ArgMaxPredictionProcessor(config={})
        # answers = self.processor["output_processor"](report)
        answers = report.scores.argmax(dim=2)
        print(answers)
        print(answers.size())

        answer = ""
        for i in range(12):
            answer += self.processor["answer_processor"].answer_vocab.idx2word(answers[0][i]) + " "
        print(type(answer))
        print(answer)
        return answer
