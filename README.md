# GliomaDeepLearning
We used the above code to predict IDH status, 1P/19Q, grade, and overall survival for gliomas in a multitask learning manner.

## 1. Requirements
   
   python >= 3.6, pytorch = 1.12.0, some packages should be installed, such as batchgenerators and lifelines.

## 2. Description of the above files
dataset_loading.py was used to generate batch data for training the prediction model, network_trainer.py and NetTrainer.py to set the training parameters and train the predictrion model, multitaskmodel1.py to set the prediction model.

Some functions might be used for trainging the model was placed in the utils.py.

The main function was saved in the run_training.py.

Cell_feature.py was used to calculate the cell features in a whole slide images.

## 3.Citations
If you find it useful for your work, please cite the following work.

Wu X, et al. Biologically interpretable multi-task deep learning pipeline predicts molecular alterations, grade, and prognosis in glioma patients. npj Precision Oncology, 2024, 8:181. DOI: 10.1038/s41698-024-00670-2
