from mmdet.models.builder import build_detector
from mmcv.utils.config import Config
from mmdet.datasets.builder import build_dataset
from mmdet.apis.train import train_detector
import mmcv
import os.path


class DetectorCreator:
    def __init__(self, config_file, classes):
        self.model = None
        self.classes = classes
        self.config = Config.fromfile(config_file)
        self.config.dataset_type = 'COCODataset'
        self.config.model.roi_head.bbox_head.num_classes = len(classes)
        self.config.model.roi_head.mask_head.num_classes = len(classes)

    def add_train_path(self, ann_file, img_prefix):
        self.config.data.train.ann_file = ann_file
        self.config.data.train.img_prefix = img_prefix
        self.config.data.train.classes = self.classes

    def add_test_path(self, ann_file, img_prefix):
        self.config.data.test.ann_file = ann_file
        self.config.data.test.img_prefix = img_prefix
        self.config.data.test.classes = self.classes

    def add_val_path(self, ann_file, img_prefix):
        self.config.data.val.ann_file = ann_file
        self.config.data.val.img_prefix = img_prefix
        self.config.data.val.classes = self.classes

    def define_checkpoint_work_dir(self, checkpoint_file, work_dir):
        self.config.load_from = checkpoint_file
        self.config.work_dir = work_dir

    def define_learning_params(self, num_gpu, num_epochs, num_interval, device):
        self.config.optimizer.lr = 0.02 / num_gpu
        self.config.lr_config.warmup = None
        self.config.gpu_ids = range(1)
        self.config.seed = 0
        self.config.device = device
        self.config.evaluation.interval = num_interval
        self.config.checkpoint_config.interval = num_interval
        self.config.runner = dict(type='EpochBasedRunner', max_epochs=num_epochs)

    def train_model(self, work_dir):
        self.model = build_detector(self.config.model)
        dataset = [build_dataset(self.config.data.train)]
        self.model.CLASSES = dataset[0].CLASSES
        mmcv.mkdir_or_exist(os.path.abspath(work_dir))
        train_detector(self.model, dataset, self.config, distributed=False, validate=True)

    def get_model(self):
        return self.model

    def get_config(self):
        return self.config
