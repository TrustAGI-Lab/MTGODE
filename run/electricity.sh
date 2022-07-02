#!/usr/bin/env bash

cd ..

python run_single_step.py \
--data ./data/electricity.txt \
--expid 0 --runs 1 --device cuda:0 --save_preds False --num_nodes 321 --horizon 3 \
--epochs 60 --batch_size 4 --lr 0.001 --weight_decay 0.0001 --lr_decay True --lr_decay_steps '20,40' --lr_decay_rate 0.5 \
--dropout 0.3 --node_dim 40 --subgraph_size 20 --num_split 1 --tanhalpha 3 --conv_channels 64 --end_channels 64 \
--solver_1 euler --time_1 1.0 --step_1 0.2 --solver_2 euler --time_2 1.0 --step_2 0.5 --alpha 1.0 --rtol 1e-4 --atol 1e-3 --adjoint False --perturb False