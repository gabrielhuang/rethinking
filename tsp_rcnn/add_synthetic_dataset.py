import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

def synthetic_dataset_func(split):
    def func():
        dataset_path = os.path.join("../d2_debug/datasets", "synthetic", split)
        image_root = dataset_path
        with open(os.path.join(dataset_path, "annotations.json")) as fin:
            annotations = json.load(fin)
        data = annotations["data"]
        for data_dict in data:
            data_dict["file_name"] = os.path.join(image_root, data_dict["file_name"])
            for obj in data_dict["annotations"]:
                obj["bbox_mode"] = BoxMode.XYXY_ABS
        return data

    return func

DatasetCatalog.register("my_synthetic_val", synthetic_dataset_func("val"))
MetadataCatalog.get("my_synthetic_val").thing_classes = ["circle", "triangle", "square"]

DatasetCatalog.register("my_synthetic_train", synthetic_dataset_func("train"))
MetadataCatalog.get("my_synthetic_train").thing_classes = ["circle", "triangle", "square"]
