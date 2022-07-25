import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from peakdetect.utils.dp_dataset import DPdataset
from peakdetect.utils.utils import worker_seed_set, load_classes
from peakdetect.utils.parse_config import parse_data_config


class DPDataModule(pl.LightningDataModule):
    def __init__(self, 
                 struct_path, 
                 class_names_path,
                 all_euler_angles_path,
                 dp_images_path_train,
                 targets_path_train,
                 euler_angles_path_train,
                 dp_images_path_valid,
                 targets_path_valid,
                 euler_angles_path_valid,
                 batch_size:int,
                 n_cpu:int,
                 pattern_size:int,
                 pattern_sigma:float,
                 reciprocal_radius:float,
                 acceleration_voltage:float,
                 max_excitation_error:float,
                 minimum_intensity:float):
        super().__init__()
        # self.args = args
        # self.data_config = parse_data_config(self.args['data'])
        # self.struct_path = self.data_config["structure"]
        self.class_names = load_classes(class_names_path)
        self.struct_path = struct_path
        self.all_euler_angles_path = all_euler_angles_path

        angle_list = list()
        with open(self.all_euler_angles_path) as f:
            for angle in f.readlines()[2:]:
                angle.replace('\n','')
                angles = angle.split()
                angle_list.append(angles)
        self.euler_angles = pd.DataFrame(angle_list, columns=['z1','x','z2']).astype(float)

        self.dp_images_path_train = dp_images_path_train
        self.targets_path_train = targets_path_train
        self.euler_angles_train = euler_angles_path_train
        self.dp_images_path_valid = dp_images_path_valid
        self.targets_path_valid = targets_path_valid
        self.euler_angles_valid = euler_angles_path_valid

        self.batch_size = batch_size
        self.n_cpu = n_cpu
        self.pattern_size = pattern_size
        self.pattern_sigma = pattern_sigma
        self.reciprocal_radius = reciprocal_radius
        self.acceleration_voltage = acceleration_voltage
        self.max_excitation_error = max_excitation_error

    def setup(self, stage = None):
        if stage == "fit" or stage is None:
            self.train = DPdataset(struct_filename=self.struct_path, 
                                   euler_angle_filename=self.euler_angles_train,
                                   dp_image_path=self.dp_images_path_train,
                                   targets_path=self.targets_path_train,
                                   pattern_size = self.pattern_size,
                                   pattern_sigma = self.pattern_sigma,
                                   reciprocal_radius = self.reciprocal_radius,
                                   acceleration_voltage=self.acceleration_voltage,
                                   max_excitation_error=self.max_excitation_error)

            self.valid = DPdataset(struct_filename=self.struct_path, 
                                   euler_angle_filename=self.euler_angles_valid,
                                   dp_image_path=self.dp_images_path_valid,
                                   targets_path=self.targets_path_valid,
                                   pattern_size = self.pattern_size,
                                   pattern_sigma = self.pattern_sigma,
                                   reciprocal_radius = self.reciprocal_radius,
                                   acceleration_voltage=self.acceleration_voltage,
                                   max_excitation_error=self.max_excitation_error)
        if stage == "test" or stage is None:
            self.test = DPdataset(struct_filename=self.struct_path, 
                                  euler_angle_filename=self.euler_angles_valid,
                                  dp_image_path=self.dp_images_path_valid,
                                  targets_path=self.targets_path_valid,
                                  pattern_size = self.pattern_size,
                                  pattern_sigma = self.pattern_sigma,
                                  reciprocal_radius = self.reciprocal_radius,
                                  acceleration_voltage=self.acceleration_voltage,
                                  max_excitation_error=self.max_excitation_error)

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.n_cpu,
                          pin_memory=True,
                          collate_fn=self.train.collate_fn,
                          worker_init_fn=worker_seed_set)
    
    def val_dataloader(self):
        return DataLoader(self.valid,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.n_cpu,
                          pin_memory=True,
                          collate_fn=self.train.collate_fn,
                          worker_init_fn=worker_seed_set)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.n_cpu,
                          pin_memory=True,
                          collate_fn=self.train.collate_fn,
                          worker_init_fn=worker_seed_set)
    def predict_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.n_cpu,
                          pin_memory=True,
                          collate_fn=self.train.collate_fn,
                          worker_init_fn=worker_seed_set)
