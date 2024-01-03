
import shutil
import traceback
from collections import OrderedDict
from multiprocessing import Pool
from time import sleep
from typing import Tuple, List
from sklearn import metrics
from lifelines.utils import concordance_index

import matplotlib
import numpy as np
import pandas as pd
import torch
from batchgenerators.utilities.file_and_folder_operations import *
from torch import nn
from torch.optim import lr_scheduler
from configuration import default_num_threads
import sys
sys.path.append("D:/work/zst/multitask/code/DL_surv/training/")
from network_training.network_trainer import NetworkTrainer
from loss_functions.focalloss import FocalLoss
from data_augmentation.default_data_augmentation import default_3D_augmentation_params, \
    default_2D_augmentation_params, get_default_augmentation, get_patch_size, get_default_augmentation_val
from dataloading.dataset_loading import load_dataset, DataLoader3D
from models.multitaskmodel1 import resnet10

matplotlib.use("agg")


def c_index(risk_pred, y, e):

    return concordance_index(y, risk_pred, e)


class NetTrainer(NetworkTrainer):
    def __init__(self, output_folder=None, dataset_directory=None, info_paths=None, deterministic=True, fp16=False):

        super(NetTrainer, self).__init__(deterministic, fp16)
        self.output_folder = output_folder
        self.dataset_directory = dataset_directory
        self.info_paths = info_paths

        self.dl_tr = self.dl_val = None

        self.loss = nn.CrossEntropyLoss()
        self.use_mask_ = None

        self.inference_pad_border_mode = "constant"
        self.inference_pad_kwargs = {'constant_values': 0}

        self.lr_scheduler_eps = 1e-3
        self.lr_scheduler_patience = 30
        self.initial_lr = 1e-3
        self.weight_decay = 3e-5

        self.oversample_foreground_percent = 0.33

        self.max_num_epochs = 500
        
        self.basic_generator_patch_size = self.data_aug_params = None
        self.batch_size = 64
        self.patch_size = np.array([88, 112, 88]).astype(int)
        self.do_dummy_2D_aug = False

        self.pad_all_sides = None
        self.normalization_schemes = None
        
        self.num_input_channels = 2
        self.num_classes = 2
        
        if len(self.patch_size) == 2:
            self.threeD = False
        elif len(self.patch_size) == 3:
            self.threeD = True
        else:
            raise RuntimeError("invalid patch size : %s" % str(self.patch_size))

    def setup_DA_params(self):

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-10. / 360 * 2. * np.pi, 10. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-10. / 360 * 2. * np.pi, 10. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-10. / 360 * 2. * np.pi, 10. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size

        self.data_aug_params["num_cached_per_thread"] = 2

    def initialize(self, training=True, force_load_plans=False):

        maybe_mkdir_p(self.output_folder)

        self.setup_DA_params()

        if training:

            self.dl_tr, self.dl_val1 = self.get_train_val_generators([self.dataset_directory[0], self.dataset_directory[1]], [self.info_paths[0], self.info_paths[1]])
            self.dl_val2 = self.get_basic_generators(self.dataset_directory[2], self.info_paths[2])
            self.dl_val3 = self.get_basic_generators(self.dataset_directory[3], self.info_paths[3])
            self.dl_val4 = self.get_basic_generators(self.dataset_directory[4], self.info_paths[4])
            
            self.tr_gen = get_default_augmentation(self.dl_tr,
                                                   self.data_aug_params[
                                                        'patch_size_for_spatialtransform'],
                                                   self.data_aug_params)
            self.val_gen1 = get_default_augmentation_val(self.dl_val1,
                                                         self.data_aug_params[
                                                             'patch_size_for_spatialtransform'],
                                                         self.data_aug_params)
            self.val_gen2 = get_default_augmentation_val(self.dl_val2,
                                                         self.data_aug_params[
                                                             'patch_size_for_spatialtransform'],
                                                         self.data_aug_params)
            self.val_gen3 = get_default_augmentation_val(self.dl_val3,
                                                          self.data_aug_params[
                                                              'patch_size_for_spatialtransform'],
                                                         self.data_aug_params)
            self.val_gen4 = get_default_augmentation_val(self.dl_val4,
                                                          self.data_aug_params[
                                                              'patch_size_for_spatialtransform'],
                                                         self.data_aug_params)
  
        else:
            pass
        self.initialize_network()
        self.initialize_optimizer_and_scheduler()
        self.was_initialized = True

    def initialize_network(self):

        self.network = resnet10(sample_input_D=self.patch_size[0], sample_input_W=self.patch_size[1], sample_input_H=self.patch_size[2], num_classes=self.num_classes)

        if torch.cuda.is_available():
            self.network.cuda()

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)

        self.lr_scheduler = None

    def plot_network_architecture(self):
        try:
            from batchgenerators.utilities.file_and_folder_operations import join
            import hiddenlayer as hl
            if torch.cuda.is_available():
                g = hl.build_graph(self.network, torch.rand((1, self.num_input_channels, *self.patch_size)).cuda(),
                                   transforms=None)
            else:
                g = hl.build_graph(self.network, torch.rand((1, self.num_input_channels, *self.patch_size)),
                                   transforms=None)
            g.save(join(self.output_folder, "network_architecture.pdf"))
            del g
        except Exception as e:
            self.print_to_log_file("Unable to plot network architecture:")
            self.print_to_log_file(e)

            self.print_to_log_file("\nprinting the network instead:\n")
            self.print_to_log_file(self.network)
            self.print_to_log_file("\n")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def run_training(self):
        super(NetTrainer, self).run_training()

    def get_train_val_generators(self, paths, info_files):

        print('loading dataset')
        
        info1 = pd.read_csv(info_files[0], encoding="gbk")
        info2 = pd.read_csv(info_files[1], encoding="gbk")
        
        np.random.seed(10)
        info1_tr = info1.sample(frac=0.7, replace=False)
        info1_val = info1[~info1.index.isin(info1_tr.index)]
        info1_tr = info1_tr.reset_index(drop=True)
        info1_val = info1_val.reset_index(drop=True)
        
        np.random.seed(10)
        info2_tr = info2.sample(frac=0.7, replace=False)
        info2_val = info2[~info2.index.isin(info2_tr.index)]
        info2_tr = info2_tr.reset_index(drop=True)
        info2_val = info2_val.reset_index(drop=True)
        
        dataset_tr = OrderedDict()
        dataset_val = OrderedDict()
        
        for c in range(info1_tr.shape[0] + info2_tr.shape[0]):  # case_identifiers
            if c < info1_tr.shape[0]:
                dataset_tr[info1_tr['Pid'][c]] = OrderedDict()
                dataset_tr[info1_tr['Pid'][c]]['data_file'] = join(paths[0], info1_tr['Pid'][c] + '.npy')
                # dataset[info['Pid'][c]]['seg_file'] = join(folder, c)
                dataset_tr[info1_tr['Pid'][c]]['info'] = [int(info1_tr['IDH'][c]), int(info1_tr['1p19q'][c]), int(info1_tr['Grade'][c]), float(info1_tr['OS'][c]), int(info1_tr['dead'][c])]
            else:
                d = c - info1_tr.shape[0]
                dataset_tr[info2_tr['Pid'][d]] = OrderedDict()
                dataset_tr[info2_tr['Pid'][d]]['data_file'] = join(paths[1], info2_tr['Pid'][d] + '.npy')
                # dataset[info['Pid'][c]]['seg_file'] = join(folder, c)
                dataset_tr[info2_tr['Pid'][d]]['info'] = [int(info2_tr['IDH'][d]), int(info2_tr['1p19q'][d]), int(info2_tr['Grade'][d]), float(info2_tr['OS'][d]), int(info2_tr['dead'][d])]
        
        for c in range(info1_val.shape[0] + info2_val.shape[0]):  # case_identifiers
            if c < info1_val.shape[0]:
                dataset_val[info1_val['Pid'][c]] = OrderedDict()
                dataset_val[info1_val['Pid'][c]]['data_file'] = join(paths[0], info1_val['Pid'][c] + '.npy')
                # dataset[info['Pid'][c]]['seg_file'] = join(folder, c)
                dataset_val[info1_val['Pid'][c]]['info'] = [int(info1_val['IDH'][c]), int(info1_val['1p19q'][c]), int(info1_val['Grade'][c]), float(info1_val['OS'][c]), int(info1_val['dead'][c])]
            else:
                d = c - info1_val.shape[0]
                dataset_val[info2_val['Pid'][d]] = OrderedDict()
                dataset_val[info2_val['Pid'][d]]['data_file'] = join(paths[1], info2_val['Pid'][d] + '.npy')
                # dataset[info['Pid'][c]]['seg_file'] = join(folder, c)
                dataset_val[info2_val['Pid'][d]]['info'] = [int(info2_val['IDH'][d]), int(info2_val['1p19q'][d]), int(info2_val['Grade'][d]), float(info2_val['OS'][d]), int(info2_val['dead'][d])]

        if self.threeD:
        
            dataloader_tr = DataLoader3D(dataset_tr, self.patch_size, self.patch_size, self.batch_size,
                                         oversample_foreground_percent=self.oversample_foreground_percent,
                                         pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='c')
    
            dataloader_val = DataLoader3D(dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                          oversample_foreground_percent=self.oversample_foreground_percent,
                                          pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='c')
    
        return dataloader_tr, dataloader_val
    
    def get_basic_generators(self, path, info_file, training=True):
        dataset = load_dataset(path, info_file)
        if self.threeD:
            if training:
                dataloader = DataLoader3D(dataset, self.patch_size, self.patch_size, self.batch_size,
                                          oversample_foreground_percent=self.oversample_foreground_percent,
                                          pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='c')
            else:
                dataloader = DataLoader3D(dataset, self.patch_size, self.patch_size, self.batch_size,
                                          oversample_foreground_percent=self.oversample_foreground_percent,
                                          pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='c')
        else:
            if training:
                dataloader = DataLoader2D(dataset, self.patch_size, self.patch_size, self.batch_size,
                                          oversample_foreground_percent=self.oversample_foreground_percent,
                                          pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='c')
            else:
                dataloader = DataLoader2D(dataset, self.patch_size, self.patch_size, self.batch_size,
                                          oversample_foreground_percent=self.oversample_foreground_percent,
                                          pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='c')
        return dataloader


    def finish_online_evaluation(self):
        
        fpr, tpr, thres = metrics.roc_curve(self.targets1_tr, self.preds1_tr)
        auc1_tr = metrics.auc(fpr, tpr)
        self.print_to_log_file("AUC Training IDH: ", str(np.round(auc1_tr, 4)))
        print("AUC Training IDH: " + str(np.round(auc1_tr, 4)))
        print("\n")

        fpr, tpr, thres = metrics.roc_curve(self.targets2_tr, self.preds2_tr)
        auc2_tr = metrics.auc(fpr, tpr)
        self.print_to_log_file("AUC Training 1p19q:", str(np.round(auc2_tr, 4)))
        print("AUC Training 1p19q:" + str(np.round(auc2_tr, 4)))
        print("\n")
        
        fpr, tpr, thres = metrics.roc_curve(self.targets3_tr, self.preds3_tr)
        auc3_tr = metrics.auc(fpr, tpr)
        self.print_to_log_file("AUC Training Grade:", str(np.round(auc3_tr, 4)))
        print("AUC Training Grade:" + str(np.round(auc3_tr, 4)))
        print("\n")
        # e = np.array(self.targets5_tr)
        
        c_ind = c_index(np.array(self.targets4_tr), -1*np.array(self.preds4_tr), np.array(self.targets5_tr))
        self.print_to_log_file("C index Training:", str(np.round(c_ind, 4)))
        print("C index Training:" + str(np.round(c_ind, 4)))
        print("\n")
        
        fpr, tpr, thres = metrics.roc_curve(self.targets1_val1, self.preds1_val1)
        auc1_val1 = metrics.auc(fpr, tpr)
        self.print_to_log_file("AUC Validation 1 IDH:", str(np.round(auc1_val1, 4)))
        print("AUC Validation 1 IDH:" + str(np.round(auc1_val1, 4)))
        print("\n")
        
        fpr, tpr, thres = metrics.roc_curve(self.targets2_val1, self.preds2_val1)
        auc2_val1 = metrics.auc(fpr, tpr)
        self.print_to_log_file("AUC Validation 1 1p19q:", str(np.round(auc2_val1, 4)))
        print("AUC Validation 1 1p19q:" + str(np.round(auc2_val1, 4)))
        print("\n")        
        
        fpr, tpr, thres = metrics.roc_curve(self.targets3_val1, self.preds3_val1)
        auc3_val1 = metrics.auc(fpr, tpr)
        self.print_to_log_file("AUC Validation 1 Grade:", str(np.round(auc3_val1, 4)))
        print("AUC Validation 1 Grade:" + str(np.round(auc3_val1, 4)))
        print("\n")
        
        # e = np.array(self.targets5_val1)
        c_ind = c_index(np.array(self.targets4_val1), -1*np.array(self.preds4_val1), np.array(self.targets5_val1))
        self.print_to_log_file("C index Validation 1:", str(np.round(c_ind, 4)))
        print("C index Validation 1:" + str(np.round(c_ind, 4)))
        print("\n")
        
        fpr, tpr, thres = metrics.roc_curve(self.targets1_val2, self.preds1_val2)
        auc1_val2 = metrics.auc(fpr, tpr)
        self.print_to_log_file("AUC Validation 2 IDH:", str(np.round(auc1_val2, 4)))
        print("AUC Validation 2 IDH:" + str(np.round(auc1_val2, 4)))
        print("\n")        
        
        fpr, tpr, thres = metrics.roc_curve(self.targets2_val2, self.preds2_val2)
        auc2_val2 = metrics.auc(fpr, tpr)
        self.print_to_log_file("AUC Validation 2 1p19q:", str(np.round(auc2_val2, 4)))
        print("AUC Validation 2 1p19q:" + str(np.round(auc2_val2, 4)))
        print("\n")        
        
        fpr, tpr, thres = metrics.roc_curve(self.targets3_val2, self.preds3_val2)
        auc3_val2 = metrics.auc(fpr, tpr)
        self.print_to_log_file("AUC Validation 2 Grade:", str(np.round(auc3_val2, 4)))
        print("AUC Validation 2 Grade:" + str(np.round(auc3_val2, 4)))
        print("\n")        
            
        c_ind = c_index(np.array(self.targets4_val2), -1*np.array(self.preds4_val2), np.array(self.targets5_val2))
        self.print_to_log_file("C index Validation 2:", str(np.round(c_ind, 4)))
        print("C index Validation 2:" + str(np.round(c_ind, 4)))
        print("\n")
                
        fpr, tpr, thres = metrics.roc_curve(self.targets1_val3, self.preds1_val3)
        auc1_val3 = metrics.auc(fpr, tpr)
        self.print_to_log_file("AUC Validation 3 IDH:", str(np.round(auc1_val3, 4)))
        print("AUC Validation 3 IDH:" + str(np.round(auc1_val3, 4)))
        print("\n")        
        
        fpr, tpr, thres = metrics.roc_curve(self.targets2_val3, self.preds2_val3)
        auc2_val3 = metrics.auc(fpr, tpr)
        self.print_to_log_file("AUC Validation 3 1p19q:", str(np.round(auc2_val3, 4)))
        print("AUC Validation 3 1p19q:" + str(np.round(auc2_val3, 4)))
        print("\n")        
        
        fpr, tpr, thres = metrics.roc_curve(self.targets3_val3, self.preds3_val3)
        auc3_val3 = metrics.auc(fpr, tpr)
        self.print_to_log_file("AUC Validation 3 Grade:", str(np.round(auc3_val3, 4)))
        print("AUC Validation 3 Grade:" + str(np.round(auc3_val3, 4)))
        print("\n")
        
        c_ind = c_index(np.array(self.targets4_val3), -1*np.array(self.preds4_val3), np.array(self.targets5_val3))
        self.print_to_log_file("C index Validation 3:", str(np.round(c_ind, 4)))
        print("C index Validation 3:" + str(np.round(c_ind, 4)))
        print("\n")
        
        fpr, tpr, thres = metrics.roc_curve(self.targets1_val4, self.preds1_val4)
        auc1_val4 = metrics.auc(fpr, tpr)
        self.print_to_log_file("AUC Validation 4 IDH:", str(np.round(auc1_val4, 4)))
        print("AUC Validation 4 IDH:" + str(np.round(auc1_val4, 4)))
        print("\n")        
        
        fpr, tpr, thres = metrics.roc_curve(self.targets2_val4, self.preds2_val4)
        auc2_val4 = metrics.auc(fpr, tpr)
        self.print_to_log_file("AUC Validation 4 1p19q:", str(np.round(auc2_val4, 4)))
        print("AUC Validation 4 1p19q:" + str(np.round(auc2_val4, 4)))
        print("\n")        
        
        fpr, tpr, thres = metrics.roc_curve(self.targets3_val4, self.preds3_val4)
        auc3_val4 = metrics.auc(fpr, tpr)
        self.print_to_log_file("AUC Validation 4 Grade:", str(np.round(auc3_val4, 4)))
        print("AUC Validation 4 Grade:" + str(np.round(auc3_val4, 4)))
        print("\n")
        

    def save_checkpoint(self, fname, save_optimizer=True):
        super(NetTrainer, self).save_checkpoint(fname, save_optimizer)


if __name__ == "__main__":
    trainer = NetTrainer()