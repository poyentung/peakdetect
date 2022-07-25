import os
import numpy as np
import pandas as pd
import pyxem as pxm
import diffpy.structure
from PIL import Image as im
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from diffsims.generators.diffraction_generator import DiffractionGenerator
from diffsims.utils.shape_factor_models import sinc

from .utils import to_cpu, worker_seed_set
from .transforms import DEFAULT_TRANSFORMS


class DPGenerator(Dataset):
    def __init__(self, 
                 struct_path, 
                 all_euler_angles_path,
                 save_folder_path = None,
                 output_name = None,
                 testset_ratio = 0.05,
                 pattern_size = 128,
                 pattern_sigma = 1.5,
                 reciprocal_radius = 2.0,
                 acceleration_voltage=200.0,
                 max_excitation_error=0.03,
                 minimum_intensity=0.1,
                 transform=DEFAULT_TRANSFORMS):
        
        self.struct_path = struct_path
        self.all_euler_angles_path = all_euler_angles_path
        self.struct = diffpy.structure.loadStructure(struct_path)
        self.save_folder_path = save_folder_path
        self.output_name = output_name
        self.testset_ratio = testset_ratio
        self.pattern_size = pattern_size
        self.pattern_sigma = pattern_sigma
        self.half_pattern_size = self.pattern_size // 2
        self.reciprocal_radius = reciprocal_radius
        self.calibration = self.reciprocal_radius / self.half_pattern_size
        self.acceleration_voltage = acceleration_voltage
        self.max_excitation_error = max_excitation_error
        self.minimum_intensity = minimum_intensity
        
        # Create dataframe for euler angles
        angle_list = list()
        with open(self.all_euler_angles_path) as f:
            for angle in f.readlines()[2:]:
                angle.replace('\n','')
                angles = angle.split()
                angle_list.append(angles)
                
        self.euler_angles = pd.DataFrame(angle_list, columns=['z1','x','z2']).astype(float)
        
        self.ediff = DiffractionGenerator(accelerating_voltage=self.acceleration_voltage, 
                                          minimum_intensity=self.minimum_intensity, 
                                          shape_factor_model=sinc)
        
        self.batch_count = 0
        self.transform = transform
    
    def __len__(self):
        return self.euler_angles.shape[0]

    def __getitem__(self, idx):
        euler_angle = self.euler_angles.loc[idx,:].to_numpy()
        ed = self.ediff.calculate_ed_data(structure = self.struct, 
                                          reciprocal_radius = self.reciprocal_radius,
                                          with_direct_beam=True,
                                          rotation=euler_angle,
                                          max_excitation_error=self.max_excitation_error,
                                          )
        ed.calibration = self.calibration
        dp_image = ed.get_diffraction_pattern(size=self.pattern_size, sigma=self.pattern_sigma)
        dp_info = self.get_dp_info(ed)
        
        # make bx, by for yolo labels
        dp_info['bx'] = dp_info['x'] / self.pattern_size
        dp_info['by'] = dp_info['y'] / self.pattern_size
        
        # make bw, bh for yolo labels
        r_avg = dp_info['intensity'].mean()
        dp_info['bw'] = np.where(dp_info['intensity']>=r_avg, 8,4) / self.pattern_size
        dp_info['bh'] = dp_info['bw']
        
        return dp_image, dp_info, euler_angle


    def get_dp_info(self, ed):
        size = self.pattern_size
        sigma = self.pattern_sigma
        side_length = np.min(np.multiply((size / 2), ed.calibration))
        mask_for_sides = np.all(
            (np.abs(ed.coordinates[:, 0:2]) < side_length), axis=1
        )

        spot_coords = np.add(
            ed.calibrated_coordinates[mask_for_sides], size / 2
        ).astype(int)

        spot_intens = ed.intensities[mask_for_sides] / ed.intensities[mask_for_sides].max()
        spot_indice = ed.indices[mask_for_sides]

        dp_info = pd.DataFrame(data={'x':spot_coords[:, 0],
                                     'y':spot_coords[:, 1], 
                                     'intensity':spot_intens,
                                     'index_h':spot_indice[:,0].astype(int),
                                     'index_k':spot_indice[:,1].astype(int),
                                     'index_l':spot_indice[:,2].astype(int)}
                        )
        
        # add one column for hkl index label
        dp_info['hkl'] = dp_info['index_h'].abs().astype(str) + dp_info['index_k'].abs().astype(str) + dp_info['index_l'].abs().astype(str)
        return dp_info
    

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


    def create_datasets(self):
        save_path_dp_image_train = os.path.join(self.save_folder_path, self.output_name+'_dp_images_train')
        save_path_targets_train = os.path.join(self.save_folder_path, self.output_name+'_targets_train')
        save_path_eulers_train = os.path.join(self.save_folder_path, self.output_name+'_euler_angles_train.csv')
        save_path_dp_image_val = os.path.join(self.save_folder_path, self.output_name+'_dp_images_val')
        save_path_targets_val = os.path.join(self.save_folder_path, self.output_name+'_targets_val')
        save_path_eulers_val = os.path.join(self.save_folder_path, self.output_name+'_euler_angles_val.csv')
        save_path_class = os.path.join(self.save_folder_path, self.output_name+'_class_names.txt')

        # make directory if not exist
        for path in (save_path_dp_image_train, save_path_targets_train, 
                     save_path_dp_image_val, save_path_targets_val):
            if not os.path.isdir(path):
                os.mkdir(path)


        self.idx_train, self.idx_val = train_test_split(range(self.__len__()),test_size=self.testset_ratio)
        self.idx_train = sorted(self.idx_train)
        self.idx_val = sorted(self.idx_val)


        print('Preparing class data ...')
        # dp_images = []
        targets={}
        # euler_angles=[]
        for idx in range(self.__len__()): #
            _, target, _ = self.__getitem__(idx)
            # dp_images.append(img)
            for hkl in target['hkl']:
                if hkl not in targets.keys():
                    targets[hkl] = 1.0
            # euler_angles.append(euler_angle)

        self.hkls = sorted(list(targets.keys())) #sorted(pd.concat(targets)['hkl'].unique())
        self.num_hkls = len(self.hkls)
        self.hkls_dict = dict()
        for i, idx in enumerate(self.hkls): self.hkls_dict[idx]=i

        with open(save_path_class, 'w+') as f:
            for hkl in self.hkls:
                f.write(hkl+'\n')
        print('Class data done!')

        # train_dp_images = []
        # train_targets=[]
        train_euler_angles=[]

        print('Preparing training data ...')
        for i, idx in enumerate(self.idx_train):
            dp_image, target, euler_angle = self.__getitem__(idx)
            target['class'] = target['hkl'].apply(lambda x: self.hkls_dict[x])
            boxes = target[['class', 'bx', 'by', 'bw', 'bh']]

            # save targets as a single .csv
            boxes.to_csv(os.path.join(save_path_targets_train,f'{i}.csv'), index=0)

            # save dp image
            im.fromarray(dp_image).save(os.path.join(save_path_dp_image_train, f'{i}.tif'))

            # train_dp_images.append(dp_image)
            # train_targets.append(boxes)
            train_euler_angles.append(euler_angle)

        print('Training data done!')

        # valid_dp_images = []
        # valid_targets=[]
        valid_euler_angles=[]

        print('Preparing validation data ...')
        for i, idx in enumerate(self.idx_val):
            dp_image, target, euler_angle = self.__getitem__(idx)
            target['class'] = target['hkl'].apply(lambda x: self.hkls_dict[x])
            boxes = target[['class', 'bx', 'by', 'bw', 'bh']]

            # save targets as a single .csv
            boxes.to_csv(os.path.join(save_path_targets_val,f'{i}.csv'), index=0)

            # save dp image
            im.fromarray(dp_image).save(os.path.join(save_path_dp_image_val, f'{i}.tif'))

            # valid_dp_images.append(dp_image)
            # valid_targets.append(boxes)
            valid_euler_angles.append(euler_angle)
        
        print('Validation data done!')

        pd.DataFrame(train_euler_angles, columns=['z1','x','z2']).astype(float).to_csv(save_path_eulers_train,index=0)
        pd.DataFrame(valid_euler_angles, columns=['z1','x','z2']).astype(float).to_csv(save_path_eulers_val,index=0)
        # return train_dp_images, train_targets, valid_dp_images, valid_targets