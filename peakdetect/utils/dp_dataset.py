import os
import torch
import numpy as np
import pandas as pd
from PIL import Image as im
import pyxem as pxm
import diffpy.structure
from torch.utils.data import Dataset, DataLoader

from diffsims.generators.diffraction_generator import DiffractionGenerator
from diffsims.utils.shape_factor_models import sinc

from .utils import to_cpu, worker_seed_set
from .transforms import DEFAULT_TRANSFORMS


# fropeakdetect.utils.dp_generatoret import DPdataset
# from sklearn.model_selection import train_test_split

# def create_dataset(struct_filename, 
#                     euler_angle_filename,
#                     pattern_size = 128,
#                     pattern_sigma = 1.5,
#                     reciprocal_radius = 2.0,
#                     acceleration_voltage=200.0,
#                     max_excitation_error=0.03,
#                     save_path=None):

#     dp_dataset = DPdataset(struct_filename,
#                             euler_angle_filename,
#                             pattern_size,
#                             pattern_sigma,
#                             reciprocal_radius,
#                             acceleration_voltage,
#                             max_excitation_error)
    
#     # make training set
#     dp_imgs = []
#     bb_targets=[]
#     for idx in range(len(dp_dataset)):
#         dp_image, bb_target, euler_angle, dp_info = dp_dataset.__getitem__(idx)
#         dp_image = dp_image.numpy()
#         bb_target = bb_target.numpy()
#         bb_target[:,0] = np.ones((bb_target.shape[0]))*idx
#         dp_imgs.append(dp_image)
#         bb_targets.append(bb_target)
    
#     dp_imgs = np.stack(dp_imgs)
#     bb_targets = np.concatenate(bb_targets,axis=0)
#     bb_targets_df = pd.DataFrame(bb_targets, columns=['id','class','bx','by','bw','bh'])
    
#     if save_path:
#         imgs_filename = euler_angle_filename.split('.')[0] + 'dp_images.npy'
#         np.save(os.path.join((save_path, imgs_filename)),dp_imgs)

#         targets_filename =  euler_angle_filename.split('.')[0] + 'targets.csv'
#         bb_targets_df.to_csv(targets_filename, index=0)
    
#     else:
#         return dp_imgs, bb_targets_df



class DPdataset(Dataset):
    def __init__(self, 
                 struct_filename, 
                 euler_angle_filename,
                 dp_image_path,
                 targets_path,
                 pattern_size = 128,
                 pattern_sigma = 1.5,
                 reciprocal_radius = 2.0,
                 acceleration_voltage=200.0,
                 max_excitation_error=0.03,
                 transform=DEFAULT_TRANSFORMS):
        
        self.struct_filename = struct_filename
        self.euler_angle_filename = euler_angle_filename
        self.struct = diffpy.structure.loadStructure(struct_filename)
        self.dp_image_path = dp_image_path
        self.targets_path = targets_path
        # self.imgs = imgs if type(imgs)==np.ndarray else np.load(imgs)
        # self.targets = targets.to_numpy() if type(targets)==pd.DataFrame else pd.read_csv(targets).to_numpy()
        self.pattern_size = float(pattern_size)
        self.pattern_sigma = float(pattern_sigma)
        self.half_pattern_size = self.pattern_size // 2
        self.reciprocal_radius = float(reciprocal_radius)
        self.calibration = self.reciprocal_radius / self.half_pattern_size
        self.acceleration_voltage = float(acceleration_voltage)
        self.max_excitation_error = float(max_excitation_error)
        
        # Create dataframe for euler angles
        self.euler_angles = pd.read_csv(self.euler_angle_filename) 
        
        # else:
        #     angle_list = list()
        #     with open(self.euler_angle_filename) as f:
        #         for angle in f.readlines()[2:]:
        #             angle.replace('\n','')
        #             angles = angle.split()
        #             angle_list.append(angles)
        #     self.euler_angles = pd.DataFrame(angle_list, columns=['z1','x','z2']).astype(float)

        self.ediff = DiffractionGenerator(accelerating_voltage=self.acceleration_voltage, shape_factor_model=sinc)
        # self.hkls = sorted(self.ediff.calculate_profile_data(self.struct,  self.reciprocal_radius).hkls)
        # self.num_hkls = len(self.hkls)
        # self.hkls_dict = dict()
        # for i, idx in enumerate(self.hkls): self.hkls_dict[idx]=i
        
        self.batch_count = 0
        self.transform = transform
    
    def __len__(self):
        return self.euler_angles.shape[0]

    def __getitem__(self, idx):
        euler_angle = self.euler_angles.to_numpy()[idx]
        
        dp_image_filename = os.path.join(self.dp_image_path, f'{idx}.tif')
        dp_image = np.array(im.open(dp_image_filename))

        targets_filename = os.path.join(self.targets_path, f'{idx}.csv')
        targets = pd.read_csv(targets_filename).to_numpy()
        # targets = self.targets[np.where(self.targets[:,0]==idx)][:,1:]
        
        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
                dp_image, bb_targets = self.transform((dp_image[:,:,np.newaxis], targets))
            except Exception:
                print("Could not apply transform.")
                return
        
        return dp_image, bb_targets, euler_angle, None
    
    
    def plot_dp(self, idx, **args):
        euler_angle = self.euler_angles.loc[idx,:].to_numpy()
        ed = self.ediff.calculate_ed_data(structure = self.struct, 
                                     reciprocal_radius = self.reciprocal_radius,
                                     with_direct_beam=True,
                                     rotation=euler_angle,
                                     max_excitation_error=self.max_excitation_error,
                                     )
        ed.calibration = self.calibration
        dp_data = ed.get_diffraction_pattern(size=self.pattern_size, sigma=self.pattern_sigma)

        dp = pxm.signals.ElectronDiffraction2D(dp_data)
        dp.set_diffraction_calibration(self.calibration)
        return dp.plot(cmap='inferno', vmax=0.33, **args)

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        dp_images, bb_targets, euler_angle, dp_info = list(zip(*batch))

#         # Selects new image size every tenth batch
#         if self.multiscale and self.batch_count % 10 == 0:
#             self.img_size = random.choice(
#                 range(self.min_size, self.max_size + 1, 32))

#         # Resize images to input shape
          # dp_images = torch.stack([resize(img, self.img_size) for dp_image in dp_images])
        
        dp_images = torch.stack(dp_images)
        
        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return dp_images, bb_targets, euler_angle, dp_info
    

# ---------- #   
# Dataloader #
# ---------- #
def _create_data_loader(struct_filename, 
                        euler_angle_filename,
                        batch_size, 
                        n_cpu,
                        imgs_path,
                        targets_path,
                        pattern_size = 128,
                        pattern_sigma = 1.5,
                        reciprocal_radius = 2.0,
                        acceleration_voltage=200.0,
                        max_excitation_error=0.03,
                        transform=DEFAULT_TRANSFORMS):
    """Creates a DataLoader for training.
    :param img_path: Path to file containing all paths to training images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :param multiscale_training: Scale images to different sizes randomly
    :type multiscale_training: bool
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = DPdataset_no_simulation(struct_filename, 
                                    euler_angle_filename,
                                    imgs_path,
                                    targets_path,
                                    pattern_size,
                                    pattern_sigma,
                                    reciprocal_radius,
                                    acceleration_voltage,
                                    max_excitation_error,
                                    transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set)
    return dataloader

def _create_validation_data_loader(struct_filename, 
                                    euler_angle_filename,
                                    batch_size, 
                                    n_cpu,
                                    imgs_path,
                                    targets_path,
                                    pattern_size = 128,
                                    pattern_sigma = 1.5,
                                    reciprocal_radius = 2.0,
                                    acceleration_voltage=200.0,
                                    max_excitation_error=0.03,
                                    transform=DEFAULT_TRANSFORMS):
    """
    Creates a DataLoader for validation.
    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = DPdataset_no_simulation(struct_filename, 
                                    euler_angle_filename,
                                    imgs_path,
                                    targets_path,
                                    pattern_size,
                                    pattern_sigma,
                                    reciprocal_radius,
                                    acceleration_voltage,
                                    max_excitation_error,
                                    transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn)
    return dataloader