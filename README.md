# EDH-Refinement
test

This file contains the implementation of evaporation ducts refinement in PyTorch.We strat from traditional super-resolution model and refactored it to meet our needs.

#How to use

The `DLCNN` derives from nn.Module so it can be used as any other PyTorch module.

Training.py is training code.Testing.py is testting code. 202001nc.nc and EDH_data.nc are eveporation duct dataset, and "tplt" and “tplb” represent eveporation duct top height and bottom height respectively.

Dataset is custom dataset code, and MODEL contain several refinement model. 

Example usage:

Model_type0 = MyModelwithAttention(Denseblock_feature_num=64, 
Denseblock_layer=3, 
Primary_extraction_receptive_field=5, 
Compensate_receptive_field=3, 
conv1_receptive_field=3, 
conv2_receptive_field=3, 
pool1_receptive_field=3, 
pool2_receptive_field=3)

Model_type1 = MyModelwithAttention_factor4(Denseblock_feature_num=64, 
Denseblock_layer=3, 
Primary_extraction_receptive_field=5, 
Compensate_receptive_field=3, 
conv1_receptive_field=3, 
conv2_receptive_field=3, 
pool1_receptive_field=3, 
pool2_receptive_field=3)

Model_type2 = MyModelwithAttention_small(Denseblock_feature_num=64, 
Denseblock_layer=3, 
Primary_extraction_receptive_field=5, 
Compensate_receptive_field=3, 
conv1_receptive_field=3, 
conv2_receptive_field=3, 
pool1_receptive_field=3, 
pool2_receptive_field=3)

Training(model_type=0, model_structure=[128, 5, 3, 3, 3, 3, 3, 3], batch_size=124, epoch=7000, mid_epoch=5000)

Testing(model_type=0, model_structure=[128, 5, 3, 3, 3, 3, 5, 3], model_path1=" ", model_path2=" ")



<!-- ![image](https://github.com/zxy1012-maker/Evaporation-Ducts-Refinement/blob/master/gitimg.png) -->




The correspondence between dataset and model is as follows：

Dataset - model

Type0 (0.5->0.25)TrainingDataset TestingDataset - MyModel.py  MyModelwithAttention.py  MyModelwithAttention_Ab_compensate.py  MyModelwithAttention_Ab_linear.py MyModelwithAttention_Ab_residual.py  MyModelwithAttention_Ab_Dense.py

Type1 (1->0.5)TrainingDataset_small - MyModelwithAttention_small.py  MyModel_small.py

Type2 (1->0.25)TrainingDataset_factor4  TestingDataset_factor4 - MyModelwithAttention_factor4.py  MyModel_factor4.py
