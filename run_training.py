import argparse
from batchgenerators.utilities.file_and_folder_operations import *
import sys
# from run.default_configuration import get_default_configuration
from load_pretrained_weights import load_pretrained_weights
sys.path.append('D:/work/zst/multitask/code/DL_surv/')
from training.network_training import NetTrainer
from utilities.to_torch import to_cuda
import pdb
import torch


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("network")
    # parser.add_argument("network_trainer")
    parser.add_argument("-val", "--validation_only", help="use this if you want to only run the validation",
                        action="store_true")
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true")

    parser.add_argument("--deterministic",
                        help="Makes training deterministic, but reduces training speed substantially. I (Fabian) think "
                             "this is not necessary. Deterministic training will make you overfit to some random seed. "
                             "Don't use that.",
                        required=False, default=False, action="store_true")
    parser.add_argument("--valbest", required=False, default=False, action="store_true",
                        help="hands off. This is not intended to be used")
    parser.add_argument("--fp32", required=False, default=False, action="store_true",
                        help="disable mixed precision training and run old school fp32")

    parser.add_argument("--disable_saving", required=False, action='store_true',
                        help="If set network will not save any parameter files (except a temporary checkpoint that "
                             "will be removed at the end of the training). Useful for development when you are "
                             "only interested in the results and want to save some disk space")
    parser.add_argument('-pretrained_weights', required=False, default=None,
                        help='path to network checkpoint file to be used as pretrained model (use .model '
                             'file, for example model_final_checkpoint.model). Will only be used when actually training. '
                             'Optional. Beta. Use with caution.')

    args = parser.parse_args()

    validation_only = args.validation_only

    deterministic = args.deterministic
    # valbest = args.valbest

    fp32 = args.fp32
    run_mixed_precision = not fp32
    output_folder_name = "E:/data/gliomas/trained_models"  # where to save results 
    dataset_directory = ["E:/data/gliomas/cropped_data/ZFH",
                         "E:/data/gliomas/cropped_data/GPH",
                         "E:/data/gliomas/cropped_data/TCGA_LGG_GBM",
                         "E:/data/gliomas/cropped_data/UCSF",
                         "E:/data/gliomas/cropped_data/EGD",]  # where the datasets saved
    info_paths = ["E:/data/gliomas/ZFH1.csv",
                  "E:/data/gliomas/GPH1.csv",
                  "E:/data/gliomas/TCGA_LGG_GBM.csv",
                  "E:/data/gliomas/UCSF-PDGM.csv",
                  "E:/data/gliomas/EGD.csv"]
    
    trainer = NetTrainer.NetTrainer(output_folder=output_folder_name, dataset_directory=dataset_directory, info_paths=info_paths, 
                         deterministic=deterministic, fp16=run_mixed_precision)
    
    if args.disable_saving:
        trainer.save_final_checkpoint = False
        trainer.save_best_checkpoint = False
        trainer.save_intermediate_checkpoints = True
        trainer.save_latest_only = True

    trainer.initialize(not validation_only)

    args.pretrained_weights = '********'
    if not validation_only:
        if args.continue_training:
            trainer.load_latest_checkpoint()
        elif (not args.continue_training) and (args.pretrained_weights is not None):
            state_dict = torch.load(args.pretrained_weights)['state_dict']
            trainer.network.load_state_dict(state_dict, strict=False)
            for name, para in trainer.network.named_parameters():
                if 'osp' in name:
                    para.requires_grad = True
                    print("trainable layer: {}".format(name))
                else:
                    para.requires_grad = False
                    print("fixable layer: {}".format(name))
        else:
            # new training without pretraine weights, do nothing
            pass
        # pdb.set_trace()
        trainer.run_training()


if __name__ == "__main__":
    main()
