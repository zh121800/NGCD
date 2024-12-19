export CUDA_VISIBLE_DEVICES=0

###################################################
# GCD
###################################################

CUDA_VISIBLE_DEVICES=2,3,4,5 python -m methods.contrastive_meanshift_training_attention  \
            --dataset_name 'cifar100' \
            --lr 0.01 \
            --temperature 0.3 \
            --wandb 


python -m methods.contrastive_meanshift_training  \
            --dataset_name 'imagenet_100' \
            --lr 0.01 \
            --temperature 0.3 \
            --wandb 

python -m methods.contrastive_meanshift_training  \
            --dataset_name 'cub' \
            --lr 0.05 \
            --temperature 0.25 \
            --eta_min 5e-3 \
            --wandb 

python -m methods.contrastive_meanshift_training  \
            --dataset_name 'scars' \
            --lr 0.05 \
            --temperature 0.25 \
            --eta_min 5e-3 \
            --wandb 

python -m methods.contrastive_meanshift_training  \
            --dataset_name 'aircraft' \
            --lr 0.05 \
            --temperature 0.25 \
            --eta_min 5e-3 \
            --wandb 

python -m methods.contrastive_meanshift_training  \
            --dataset_name 'herbarium_19' \
            --lr 0.05 \
            --temperature 0.25 \
            --eta_min 5e-3 \
            --wandb 


###################################################
# INDUCTIVE GCD
###################################################

python -m methods.contrastive_meanshift_training  \
            --dataset_name 'cifar100' \
            --lr 0.01 \
            --temperature 0.3 \
            --inductive \
            --wandb 

python -m methods.contrastive_meanshift_training  \
            --dataset_name 'imagenet_100' \
            --lr 0.01 \
            --temperature 0.3 \
            --inductive \
            --wandb 

python -m methods.contrastive_meanshift_training  \
            --dataset_name 'cub' \
            --lr 0.05 \
            --temperature 0.25 \
            --eta_min 5e-3 \
            --inductive \
            --wandb 

python -m methods.contrastive_meanshift_training  \
            --dataset_name 'scars' \
            --lr 0.05 \
            --temperature 0.25 \
            --eta_min 5e-3 \
            --inductive \
            --wandb 

python -m methods.contrastive_meanshift_training  \
            --dataset_name 'aircraft' \
            --lr 0.05 \
            --temperature 0.25 \
            --eta_min 5e-3 \
            --inductive \
            --wandb 

python -m methods.contrastive_meanshift_training  \
            --dataset_name 'herbarium_19' \
            --lr 0.05 \
            --temperature 0.25 \
            --eta_min 5e-3 \
            --inductive \
            --wandb 