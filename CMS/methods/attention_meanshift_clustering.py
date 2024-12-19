import wandb
import os
import torch
import timm
import argparse
import numpy as np


from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from data.stanford_cars import CarsDataset
from data.cifar import CustomCIFAR10, CustomCIFAR100, cifar_10_root, cifar_100_root
from data.herbarium_19 import HerbariumDataset19, herbarium_dataroot
from data.augmentations import get_transform
from data.imagenet import get_imagenet_100_datasets
from data.data_utils import MergedDataset
from data.cub import CustomCub2011, cub_root, get_cub_datasets
from data.fgvc_aircraft import FGVCAircraft, aircraft_root
from data.get_datasets import get_datasets, get_class_splits, get_datasets_with_gcdval

from project_utils.general_utils import str2bool
from copy import deepcopy
from project_utils.cluster_and_log_utils import *

from models.dino import *
from models import vision_transformer as vits
from config import dino_pretrain_path

def iterative_meanshift(model, loader, args):
    """
    This function measures clustering accuracies on GCD setup in both GT, predicted number of classes.

    Clustering : labeled train dataset, unlabeled train dataset.
    Stopping condition : labeled train dataset.
    Acc : unlabeled train dataset.
    """
    # Initialize variables
    num_clusters = [args.num_labeled_classes + args.num_unlabeled_classes, args.num_clusters]
    acc = [0, 0]
    max_acc = [0, 0]
    tolerance = [0, 0]
    final_acc = [(0, 0, 0), (0, 0, 0)]
    print('Predicted number of clusters: ', args.num_clusters)

    # Initialize feature storage and targets
# Initialize feature storage and targets
    all_feats = torch.zeros(size=(len(loader.dataset), args.feat_dim), device="cpu")
    new_feats = torch.zeros(size=(len(loader.dataset), args.feat_dim), device="cpu")
    targets = torch.zeros(len(loader.dataset), dtype=torch.long, device="cpu")  # 改为 long 类型
    mask_lab = torch.zeros(len(loader.dataset), dtype=torch.bool, device="cpu")
    mask_cls = torch.zeros(len(loader.dataset), dtype=torch.bool, device="cpu")


    with torch.no_grad():
        for epoch in range(args.epochs):
            print(f"Epoch {epoch + 1}/{args.epochs}")
            
            # Save embeddings
            for batch_idx, batch in enumerate(tqdm(loader)):
                images, label, uq_idxs, mask_lab_ = batch
                images = images.to(device)  # Move images to computation device

                if epoch == 0:
                    # Extract initial embeddings
                    all_feats[uq_idxs] = model(images).detach().cpu()
                    targets[uq_idxs] = label.cpu()
                    mask_lab[uq_idxs] = mask_lab_.squeeze(1).cpu().bool()
                else:
                    # Normalize all_feats before computing similarity
                    all_feats = F.normalize(all_feats, dim=-1)
                    uq_feats = F.normalize(all_feats[uq_idxs], dim=-1)

                    # Compute pairwise similarity
                    classwise_sim = torch.einsum('b d, n d -> b n', uq_feats, all_feats)
                    _, indices = classwise_sim.topk(k=args.k + 1, dim=-1, largest=True, sorted=True)
                    indices = indices[:, 1:]  # Exclude self

                    # Extract kNN features and normalize
                    knn_feats = all_feats[indices]
                    knn_feats = F.normalize(knn_feats, dim=-1)

                    # Attention mechanism for weighted kNN features
                    Q = uq_feats.unsqueeze(1)  # Query: (batch_size, 1, feature_dim)
                    K = knn_feats  # Key: (batch_size, k, feature_dim)
                    V = knn_feats  # Value: (batch_size, k, feature_dim)

                    # Attention weights
                    d_k = Q.size(-1)  # Feature dimension
                    attention_logits = torch.einsum('b q d, b k d -> b q k', Q, K) / (d_k**0.5)
                    attention_weights = torch.softmax(attention_logits + 1e-6, dim=-1)

                    # Compute weighted features
                    weighted_knn_feats = torch.einsum('b q k, b k d -> b q d', attention_weights, V)
                    weighted_knn_feats = weighted_knn_feats.squeeze(1)  # (batch_size, feature_dim)

                    # Update features
                    new_feats[uq_idxs] = (1 - args.alpha) * all_feats[uq_idxs] + args.alpha * weighted_knn_feats.detach().cpu()

            if epoch == 0:
                # Split targets into labeled and unlabeled
                mask_cls = np.isin(targets, range(len(args.train_classes))).astype(bool)
                mask_lab = mask_lab.numpy().astype(bool)
                l_targets = targets[mask_lab].numpy()
                u_targets = targets[~mask_lab].numpy()
                mask = mask_cls[~mask_lab]
                mask = mask.astype(bool)
            else:
                # Normalize updated features
                new_feats = F.normalize(new_feats, dim=-1)
                all_feats = new_feats

            # Perform clustering
            linked = linkage(all_feats.numpy(), method="ward")
            for i in range(len(num_clusters)):
                if num_clusters[i]:
                    print('Clustering with', num_clusters[i], 'clusters...')
                else:
                    continue

                # Set threshold for clustering
                threshold = linked[:, 2][-num_clusters[i]]
                preds = fcluster(linked, t=threshold, criterion='distance')

                # Compute accuracy
                old_acc_train = compute_acc(l_targets, preds[mask_lab])
                all_acc_test, old_acc_test, new_acc_test = log_accs_from_preds(
                    y_true=u_targets,
                    y_pred=preds[~mask_lab],
                    mask=mask,
                    T=epoch,
                    eval_funcs=args.eval_funcs,
                    save_name='IMS unlabeled train ACC',
                    print_output=True,
                )

                # Stopping condition
                tolerance[i] = 0 if max_acc[i] < old_acc_train else tolerance[i] + 1
                if max_acc[i] <= old_acc_train:
                    max_acc[i] = old_acc_train
                    acc[i] = (all_acc_test, old_acc_test, new_acc_test)

                # Stop if tolerance exceeded
                if tolerance[i] >= 4:
                    num_clusters[i] = 0
                    final_acc[i] = acc[i]

            # Break if both GT & predicted clusters stopped
            if sum(num_clusters) == 0:
                break

        # Final results
        print(f'ACC with GT number of clusters: All {final_acc[0][0]:.4f} | Old {final_acc[0][1]:.4f} | New {final_acc[0][2]:.4f}')
        print(f'ACC with predicted number of clusters: All {final_acc[1][0]:.4f} | Old {final_acc[1][1]:.4f} | New {final_acc[1][2]:.4f}')

        # Log final results to Wandb
        wandb.log({
            'final_GT_clusters_all_acc': final_acc[0][0],
            'final_GT_clusters_old_acc': final_acc[0][1],
            'final_GT_clusters_new_acc': final_acc[0][2],
            'final_Predicted_clusters_all_acc': final_acc[1][0],
            'final_Predicted_clusters_old_acc': final_acc[1][1],
            'final_Predicted_clusters_new_acc': final_acc[1][2],
        })

def iterative_meanshift_inductive(model, loader, val_loader, args):
    """
    This function measures clustering accuracies on inductive GCD setup in both GT, predicted number of classes
    Clustering : test dataset
    Stopping condition : val dataset
    Acc : test dataset
    """

    num_clusters = [args.num_labeled_classes + args.num_unlabeled_classes, args.num_clusters]
    acc = [0, 0]
    max_acc = [0, 0]
    tolerance = [0, 0]
    final_acc = [(0, 0, 0), (0, 0, 0)]
    print('Predicted number of clusters: ', args.num_clusters)

    all_feats = torch.zeros(size=(len(loader.dataset), args.feat_dim))
    new_feats = torch.zeros(size=(len(loader.dataset), args.feat_dim))
    targets = torch.zeros(len(loader.dataset), dtype=int)
    mask_cls = torch.zeros(len(loader.dataset), dtype=bool)

    all_feats_val = []
    new_feats_val = []
    targets_val = []
    mask_cls_val = []

    with torch.no_grad():
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}:")

            # -------------------------------
            # Save embeddings (test dataset)
            # -------------------------------
            print("Saving embeddings for test dataset...")
            for batch_idx, batch in enumerate(tqdm(loader)):
                images, label, uq_idxs = batch
                if epoch == 0:
                    images = torch.Tensor(images).to(device)
                    all_feats[uq_idxs] = model(images).detach().cpu()
                    targets[uq_idxs] = label.cpu()
                else:
                    # 计算当前批次与所有测试样本的相似度
                    classwise_sim = torch.einsum('b d, n d -> b n', all_feats[uq_idxs], all_feats)
                    _, indices = classwise_sim.topk(k=args.k + 1, dim=-1, largest=True, sorted=True)
                    indices = indices[:, 1:]  # 排除自身

                    # 提取 kNN 特征
                    knn_feats = all_feats[indices]

                    # 注意力机制计算加权 kNN 特征
                    Q = all_feats[uq_idxs].unsqueeze(1)  # Query: (batch_size, 1, feature_dim)
                    K = knn_feats  # Key: (batch_size, k, feature_dim)
                    V = knn_feats  # Value: (batch_size, k, feature_dim)

                    # 注意力权重：点积 + 缩放 + Softmax
                    d_k = Q.size(-1)
                    attention_logits = torch.einsum('b q d, b k d -> b q k', Q, K) / (d_k**0.5)
                    attention_weights = torch.softmax(attention_logits, dim=-1)

                    # 加权求和
                    weighted_knn_feats = torch.einsum('b q k, b k d -> b q d', attention_weights, V)
                    weighted_knn_feats = weighted_knn_feats.squeeze(1)

                    # 更新特征
                    new_feats[uq_idxs] = (1 - args.alpha) * all_feats[uq_idxs] + args.alpha * weighted_knn_feats.detach().cpu()

            if epoch == 0:
                mask_cls = np.isin(targets, range(len(args.train_classes))).astype(bool)
                targets = np.array(targets)
            else:
                norm = torch.sqrt(torch.sum((torch.pow(new_feats, 2)), dim=-1)).unsqueeze(1)
                new_feats = new_feats / norm
                all_feats = new_feats

            # -------------------------------
            # Save embeddings (validation dataset)
            # -------------------------------
            print("Saving embeddings for validation dataset...")
            for batch_idx, batch in enumerate(tqdm(val_loader)):
                images, label, uq_idxs = batch[:3]
                if epoch == 0:
                    images = torch.Tensor(images).to(device)
                    all_feats_val.append(model(images).detach().cpu())
                    targets_val.append(label.cpu())
                else:
                    start_idx = batch_idx * args.batch_size
                    classwise_sim = torch.einsum('b d, n d -> b n', all_feats_val[start_idx:start_idx + len(uq_idxs)], all_feats_val)
                    _, indices = classwise_sim.topk(k=args.k + 1, dim=-1, largest=True, sorted=True)
                    indices = indices[:, 1:]

                    # 提取 kNN 特征
                    knn_feats_val = all_feats_val[indices]

                    # 注意力机制计算加权 kNN 特征
                    Q = all_feats_val[start_idx:start_idx + len(uq_idxs)].unsqueeze(1)
                    K = knn_feats_val
                    V = knn_feats_val

                    d_k = Q.size(-1)
                    attention_logits = torch.einsum('b q d, b k d -> b q k', Q, K) / (d_k**0.5)
                    attention_weights = torch.softmax(attention_logits, dim=-1)
                    weighted_knn_feats_val = torch.einsum('b q k, b k d -> b q d', attention_weights, V).squeeze(1)

                    new_feats_val[start_idx:start_idx + len(uq_idxs)] = (
                        (1 - args.alpha) * all_feats_val[start_idx:start_idx + len(uq_idxs)] + args.alpha * weighted_knn_feats_val.detach().cpu()
                    )

            if epoch == 0:
                all_feats_val = torch.cat(all_feats_val)
                targets_val = np.array(torch.cat(targets_val))
                mask_cls_val = np.isin(targets_val, range(len(args.train_classes))).astype(bool)
                new_feats_val = all_feats_val
            else:
                norm = torch.sqrt(torch.sum((torch.pow(new_feats_val, 2)), dim=-1)).unsqueeze(1)
                new_feats_val = new_feats_val / norm
                all_feats_val = new_feats_val

            # -------------------------------
            # Agglomerative clustering
            # -------------------------------
            linked = linkage(all_feats, method="ward")
            linked_val = linkage(all_feats_val, method="ward")
            for i in range(len(num_clusters)):
                if num_clusters[i]:
                    print('num clusters', num_clusters[i])
                else:
                    continue
                print(f"Clustering with {num_clusters[i]} clusters...")

                # Validation accuracy
                threshold_val = linked_val[:, 2][-num_clusters[i]]
                preds_val = fcluster(linked_val, t=threshold_val, criterion='distance')
                old_acc_val = compute_acc(targets_val[mask_cls_val], preds_val[mask_cls_val])

                # Test accuracy
                threshold_test = linked[:, 2][-num_clusters[i]]
                preds_test = fcluster(linked, t=threshold_test, criterion='distance')
                all_acc_test, old_acc_test, new_acc_test = log_accs_from_preds(
                    y_true=targets, y_pred=preds_test, mask=mask_cls,
                    T=epoch, eval_funcs=args.eval_funcs, save_name='IMS test ACC', print_output=True
                )



                # Stopping condition
                tolerance[i] = 0 if max_acc[i] < old_acc_val else tolerance[i] + 1
                if max_acc[i] <= old_acc_val:
                    max_acc[i] = old_acc_val
                    acc[i] = (all_acc_test, old_acc_test, new_acc_test)

                if tolerance[i] >= 2:
                    num_clusters[i] = 0
                    final_acc[i] = acc[i]

            # If all clusters stopped, exit
            if sum(num_clusters) == 0:
                break

        # -------------------------------
        # Final results
        # -------------------------------
        print(f'ACC with GT number of clusters: All {final_acc[0][0]:.4f} | Old {final_acc[0][1]:.4f} | New {final_acc[0][2]:.4f}')
        print(f'ACC with predicted number of clusters: All {final_acc[1][0]:.4f} | Old {final_acc[1][1]:.4f} | New {final_acc[1][2]:.4f}')

        wandb.log({
            'final_GT_clusters_all_acc': final_acc[0][0],
            'final_GT_clusters_old_acc': final_acc[0][1],
            'final_GT_clusters_new_acc': final_acc[0][2],
            'final_pred_clusters_all_acc': final_acc[1][0],
            'final_pred_clusters_old_acc': final_acc[1][1],
            'final_pred_clusters_new_acc': final_acc[1][2],
        })
def compute_acc(y_true, y_pred):
    y_true = y_true.astype(int)
    old_classes_gt = set(y_true)

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # w: pred x label count
    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    return total_acc




if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='cluster',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pretrain_path', type=str, default='./models')
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--eval_funcs', type=str, default=['v2'])
    parser.add_argument('--use_ssb_splits', type=str2bool, default=True)
    parser.add_argument('--model_name', type=str, default='cms', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='aircraft', help='options: cifar10, cifar100, scars')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--feat_dim', default=768, type=int)
    parser.add_argument('--num_clusters', default=None, type=int)
    parser.add_argument('--inductive', action='store_true')
    parser.add_argument('--k', default=8, type=int)
    parser.add_argument('--alpha', type=float, default=0.5)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    print(args)
    args.model_name = "/home/beichen_zhang/zhenghao/cl/CMS/log_attention/cms/log/(18.12.2024_|_16.391)/checkpoints/model_best.pt"
    print(f'Using weights from {args.model_name} ...')

    # ----------------------
    # Initialize wandb
    # ----------------------
    wandb.init(
        project="CMS",  # 项目名称
        name=f"experiment_{args.dataset_name}_k{args.k}_alpha{args.alpha}",  # 给实验一个唯一的名称
        config=args, # 自动记录 argparse 配置
        mode="offline"
    )
    wandb.log({
        'model_path': args.model_name,
        'dataset_name': args.dataset_name,
        'num_clusters': args.num_clusters,
        'batch_size': args.batch_size,
        'epochs': args.epochs
    })

    # ----------------------
    # MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875
    args.prop_train_labels = 0.5
    args.num_mlp_layers = 3
    args.feat_dim = 768

    model = DINO(args)
    model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    state_dict = torch.load(args.model_name)
    model.load_state_dict(state_dict['model_state_dict'], strict=False)
    args.num_clusters = state_dict['k']

    # ----------------------
    # DATASET
    # ----------------------
    train_transform, test_transform = get_transform(args.transform, image_size=224, args=args)
    if args.inductive:
        _, test_dataset, _, val_dataset, _ = get_datasets_with_gcdval(args.dataset_name, test_transform, test_transform, args)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        # Run inductive clustering
        iterative_meanshift_inductive(model, test_loader, val_loader, args)

    else:
        train_dataset, _, _, _ = get_datasets(args.dataset_name, test_transform, test_transform, args)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        # Run clustering
        iterative_meanshift(model, train_loader, args)


    # ----------------------
    # Finish wandb logging
    # ----------------------
    wandb.finish()

