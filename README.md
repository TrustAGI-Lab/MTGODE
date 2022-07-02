# MTGODE

[![Arxiv link](https://img.shields.io/static/v1?label=arXiv&message=MTGODE&color=blue&logo=arxiv)](https://arxiv.org/pdf/2202.08408.pdf)

This is the official PyTorch implementation of the paper "[Multivariate Time Series Forecasting with Dynamic Graph Neural ODEs](https://arxiv.org/pdf/2202.08408.pdf)".

## Dependencies

```
numpy==1.19.2
scipy==1.5.4
torch==1.7.1
torchdiffeq==0.2.3
```
To install all dependencies:
```
pip install -r requirements.txt
```

## Download datasets

+ You can download the datasets from [here](https://drive.google.com/drive/folders/1dPy46cXUO_fjKSLqa1mWtkwbkGsMO4PG?usp=sharing).

+ Please put all dataset files under the ```./data``` directory.

## Reproducibility
Here we provide two examples (i.e., METR-LA and Electricity)
### In terminal
+ Option 1. Run the shell scripts (i.e., ```eleltricity.sh``` and ```metr-la.sh```)
```
cd run
bash metr-la.sh
```
+ Option 2. Run the python files
  + To run on METR-LA:
  ```
    python run_multi_step.py --data ./data/METR-LA --buildA_true --expid 0 --runs 1 --device cuda:6 --save_preds False --num_nodes 207 --epochs 200 --batch_size 64 --learning_rate 0.001 --weight_decay 0.0001 --lr_decay True --lr_decay_steps 100 --lr_decay_rate 0.1 --dropout 0.3 --node_dim 40 --subgraph_size 20 --num_split 1 --tanhalpha 3 --conv_channels 64 --end_channels 128 --solver_1 euler --time_1 1.0 --step_1 0.25 --solver_2 euler --time_2 1.0 --step_2 0.25 --alpha 2.0 --rtol 1e-4 --atol 1e-3 --adjoint False --perturb False
   ```
  + To run on Electricity (horizon=3):

### Results
Here we provide the results of the above two examples
+ METR-LA
```
Training finished
The valid loss on best model is 2.6909
Evaluate best model on test data for horizon 1, Test MAE: 2.2302, Test MAPE: 0.0539, Test RMSE: 3.8937
Evaluate best model on test data for horizon 2, Test MAE: 2.4785, Test MAPE: 0.0620, Test RMSE: 4.6105
Evaluate best model on test data for horizon 3, Test MAE: 2.6491, Test MAPE: 0.0682, Test RMSE: 5.0816
Evaluate best model on test data for horizon 4, Test MAE: 2.7805, Test MAPE: 0.0732, Test RMSE: 5.4578
Evaluate best model on test data for horizon 5, Test MAE: 2.8886, Test MAPE: 0.0775, Test RMSE: 5.7657
Evaluate best model on test data for horizon 6, Test MAE: 2.9818, Test MAPE: 0.0812, Test RMSE: 6.0234
Evaluate best model on test data for horizon 7, Test MAE: 3.0659, Test MAPE: 0.0845, Test RMSE: 6.2481
Evaluate best model on test data for horizon 8, Test MAE: 3.1410, Test MAPE: 0.0873, Test RMSE: 6.4417
Evaluate best model on test data for horizon 9, Test MAE: 3.2070, Test MAPE: 0.0899, Test RMSE: 6.6087
Evaluate best model on test data for horizon 10, Test MAE: 3.2681, Test MAPE: 0.0924, Test RMSE: 6.7568
Evaluate best model on test data for horizon 11, Test MAE: 3.3243, Test MAPE: 0.0946, Test RMSE: 6.8920
Evaluate best model on test data for horizon 12, Test MAE: 3.3835, Test MAPE: 0.0968, Test RMSE: 7.0258
```
+ Electricity (horizon=3)
```
final test rse 0.0728 | test rae 0.0415 | test corr 0.9431
```

## Cite us
```
@article{jin2022multivariate,
  title={Multivariate Time Series Forecasting with Dynamic Graph Neural ODEs},
  author={Jin, Ming and Zheng, Yu and Li, Yuan-Fang and Chen, Siheng and Yang, Bin and Pan, Shirui},
  journal={arXiv preprint arXiv:2202.08408},
  year={2022}
}
```

## Acknowledgement
Our implementation adapts the code [here](https://github.com/nnzhan/MTGNN) as the code base and extensively adapts it to our purpose. We thank the authors for sharing their code.