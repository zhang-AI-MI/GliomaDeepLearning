import pandas as pd
from collections import OrderedDict
import numpy as np
from multiprocessing import Pool

from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.utilities.file_and_folder_operations import *


def convert_to_npy(args):
    if not isinstance(args, tuple):
        key = "data"
        npz_file = args
    else:
        npz_file, key = args
    if not isfile(npz_file[:-3] + "npy"):
        a = np.load(npz_file)[key]
        np.save(npz_file[:-3] + "npy", a)


def save_as_npz(args):
    if not isinstance(args, tuple):
        key = "data"
        npy_file = args
    else:
        npy_file, key = args
    d = np.load(npy_file)
    np.savez_compressed(npy_file[:-3] + "npz", **{key: d})


def unpack_dataset(folder, threads=2, key="data"):
    """
    unpacks all npz files in a folder to npy (whatever you want to have unpacked must be saved unter key)
    :param folder:
    :param threads:
    :param key:
    :return:
    """
    p = Pool(threads)
    npz_files = subfiles(folder, True, None, ".npz", True)
    p.map(convert_to_npy, zip(npz_files, [key] * len(npz_files)))
    p.close()
    p.join()


def pack_dataset(folder, threads=2, key="data"):
    p = Pool(threads)
    npy_files = subfiles(folder, True, None, ".npy", True)
    p.map(save_as_npz, zip(npy_files, [key] * len(npy_files)))
    p.close()
    p.join()


def delete_npy(folder):
    case_identifiers = get_case_identifiers(folder)
    npy_files = [join(folder, i + ".npy") for i in case_identifiers]
    npy_files = [i for i in npy_files if isfile(i)]
    for n in npy_files:
        os.remove(n)


def load_dataset(folder, info_file):
    # we don't load the actual data but instead return the filename to the np file.
    print('loading dataset')
    
    info = pd.read_csv(info_file, encoding="gbk")
    dataset = OrderedDict()
    for c in range(info.shape[0]):  # case_identifiers
        dataset[info['Pid'][c]] = OrderedDict()
        dataset[info['Pid'][c]]['data_file'] = join(folder, info['Pid'][c] + '.npy')
        
        dataset[info['Pid'][c]]['info'] = [int(info['IDH'][c]), int(info['1p19q'][c]), int(info['Grade'][c]), float(info['OS'][c]), int(info['dead'][c])]
                        
    return dataset


def crop_2D_image_force_fg(img, crop_size, valid_voxels):
    """
    img must be [c, x, y]
    img[-1] must be the segmentation with segmentation>0 being foreground
    :param img:
    :param crop_size:
    :param valid_voxels: voxels belonging to the selected class
    :return:
    """
    assert len(valid_voxels.shape) == 2

    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size] * (len(img.shape) - 1)
    else:
        assert len(crop_size) == (len(
            img.shape) - 1), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (3d)"

    # we need to find the center coords that we can crop to without exceeding the image border
    lb_x = crop_size[0] // 2
    ub_x = img.shape[1] - crop_size[0] // 2 - crop_size[0] % 2
    lb_y = crop_size[1] // 2
    ub_y = img.shape[2] - crop_size[1] // 2 - crop_size[1] % 2

    if len(valid_voxels) == 0:
        selected_center_voxel = (np.random.random_integers(lb_x, ub_x),
                                 np.random.random_integers(lb_y, ub_y))
    else:
        selected_center_voxel = valid_voxels[np.random.choice(valid_voxels.shape[1]), :]

    selected_center_voxel = np.array(selected_center_voxel)
    for i in range(2):
        selected_center_voxel[i] = max(crop_size[i] // 2, selected_center_voxel[i])
        selected_center_voxel[i] = min(img.shape[i + 1] - crop_size[i] // 2 - crop_size[i] % 2,
                                       selected_center_voxel[i])

    result = img[:, (selected_center_voxel[0] - crop_size[0] // 2):(
            selected_center_voxel[0] + crop_size[0] // 2 + crop_size[0] % 2),
             (selected_center_voxel[1] - crop_size[1] // 2):(
                     selected_center_voxel[1] + crop_size[1] // 2 + crop_size[1] % 2)]
    return result


class DataLoader3D(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, num_seg = 1,
                 oversample_foreground_percent=0.0, memmap_mode="c", pad_mode="edge", pad_kwargs_data=None,
                 pad_sides=None):
        
        super(DataLoader3D, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.num_seg = num_seg
        self.list_of_keys = list(self._data.keys())
        # need_to_pad denotes by how much we need to pad the data so that if we sample a patch of size final_patch_size
        # (which is what the network will get) these patches will also cover the border of the patients
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.memmap_mode = memmap_mode
        self.num_channels = None
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.start_id = 0  # for predict
        self.end_id = self.start_id + self.batch_size  # for predict
        self.end = len(self.list_of_keys)

    def determine_shapes(self):
        k = list(self._data.keys())[0]
        case_all_data = np.load(self._data[k]['data_file'], self.memmap_mode)

        num_color_channels = case_all_data.shape[0] - 1
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, self.num_seg, *self.patch_size)
        return data_shape, seg_shape

    def generate_train_batch(self):
        # 'for training'
        # selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)  # for Train
        
        # 'for inference'
        if self.start_id >= self.end:
            raise StopIteration
        selected_keys = []
        
        for i in range(self.start_id, self.end_id):
            selected_keys.append(self.list_of_keys[i])
        # 'for inference end'
        
        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)
        
        # 'for inference'
        if self.end_id < self.start_id + self.batch_size:
            data = np.zeros((self.end_id - self.start_id, data.shape[1], data.shape[2], data.shape[3], data.shape[4]), dtype=np.float32)
            seg = np.zeros((self.end_id - self.start_id, 1, data.shape[2], data.shape[3], data.shape[4]), dtype=np.float32)
            
        self.start_id = self.start_id + self.batch_size
        
        self.end_id = min(self.end, self.end_id + self.batch_size)
        # 'for inference end'
        
        case_properties = []
        for j, i in enumerate(selected_keys):
            
            if 'info' in self._data[i].keys():
                properties = self._data[i]['info']
            else:
                raise Exception('infomation is not available for ' + self._data[i]['data_file'])
            case_properties.append(properties)
            
            if isfile(self._data[i]['data_file']):
                case_all_data = np.load(self._data[i]['data_file'], self.memmap_mode)
                case_all_data[-1][case_all_data[-1, :, :, :] > 2] = 1
                
            else:
                raise Exception("t1c.npy or flair.npy dose not exist for " + self._data[i]['data_file'])

            need_to_pad = self.need_to_pad.copy()
            for d in range(3):
                if need_to_pad[d] + case_all_data.shape[d + 1] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d + 1]

            shape = case_all_data.shape[1:]
            lb_x = - need_to_pad[0] // 2
            ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
            lb_y = - need_to_pad[1] // 2
            ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]
            lb_z = - need_to_pad[2] // 2
            ub_z = shape[2] + need_to_pad[2] // 2 + need_to_pad[2] % 2 - self.patch_size[2]

            bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
            bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
            bbox_z_lb = np.random.randint(lb_z, ub_z + 1)
            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]
            bbox_z_ub = bbox_z_lb + self.patch_size[2]

            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)
            valid_bbox_z_lb = max(0, bbox_z_lb)
            valid_bbox_z_ub = min(shape[2], bbox_z_ub)

            case_all_data = np.copy(case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                                    valid_bbox_y_lb:valid_bbox_y_ub,
                                    valid_bbox_z_lb:valid_bbox_z_ub])
            
            data[j] = np.pad(case_all_data[:-1], ((0, 0),
                                                  (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                  (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                  (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                             self.pad_mode, **self.pad_kwargs_data)
            seg[j, 0] = np.pad(case_all_data[-1:], ((0, 0),
                                                    (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                    (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                    (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                               'constant', **{'constant_values': -1})
            
        return {'data': data, 'seg': seg, 'properties': case_properties, 'keys': selected_keys}
