#!/usr/bin/env bash

cd ..

python run_multi_step.py \
--data data/METR-LA --buildA_true True \
--expid 0 --runs 1 --device cuda:0 --save_preds False --num_nodes 207 \
--epochs 200 --batch_size 64 --learning_rate 0.001 --weight_decay 0.0001 --lr_decay True --lr_decay_steps 100 --lr_decay_rate 0.1 --dropout 0.3 \
--node_dim 40 --subgraph_size 20 --num_split 1 --tanhalpha 3 --conv_channels 64 --end_channels 128  \
--solver_1 euler --time_1 1.0 --step_1 0.25 --solver_2 euler --time_2 1.0 --step_2 0.25 --alpha 2.0 --rtol 1e-4 --atol 1e-3 --adjoint False --perturb False