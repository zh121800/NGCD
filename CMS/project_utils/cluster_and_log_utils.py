from project_utils.cluster_utils import cluster_acc, np, linear_assignment
from torch.utils.tensorboard import SummaryWriter
from typing import List
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
#v1是不知道gt
#v2知道gt标签使用匈牙利算法分配标签。
def split_cluster_acc_v1(y_true, y_pred, mask):

    """
    Evaluate clustering metrics on two subsets of data, as defined by the mask 'mask'
    (Mask usually corresponding to `Old' and `New' classes in GCD setting)
    :param targets: All ground truth labels
    :param preds: All predictions
    :param mask: Mask defining two subsets
    :return:
    """

    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    weight = mask.mean()
    old_acc = cluster_acc(y_true[mask], y_pred[mask])
    new_acc = cluster_acc(y_true[~mask], y_pred[~mask])
    total_acc = weight * old_acc + (1 - weight) * new_acc

    return total_acc, old_acc, new_acc

def split_cluster_acc_v2(y_true, y_pred, mask):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    # D为总类别
    D = max(y_pred.max(), y_true.max()) + 1
    # W为权重矩阵
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # w: pred x label count
    ind = linear_assignment(w.max() - w)
    #np.vstack(ind) 将两个数组堆叠成二维数组
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    old_acc = 0
    #每个类别的准确率
    classwise_acc = []
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
        classwise_acc.append(w[ind_map[i], i]/sum(w[:, i]))
    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]         # w[pred matching with ith label, ith label]
        total_new_instances += sum(w[:, i])
        classwise_acc.append(w[ind_map[i], i]/sum(w[:, i]))
    new_acc /= total_new_instances

    return total_acc, old_acc, new_acc

#权重矩阵 w[i, j] 表示预测标签为 i 的样本被分配到真实标签为 j 的样本数量。
EVAL_FUNCS = {
    'v1': split_cluster_acc_v1,
    'v2': split_cluster_acc_v2,
}
#v1对应gt
def log_accs_from_preds(y_true, y_pred, mask, eval_funcs: List[str], save_name: str, T: int=None, writer: SummaryWriter=None,
                        print_output=False):

    """
    Given a list of evaluation functions to use (e.g ['v1', 'v2']) evaluate and log ACC results

    :param y_true: GT labels
    :param y_pred: Predicted indices
    :param mask: Which instances belong to Old and New classes
    :param T: Epoch
    :param eval_funcs: Which evaluation functions to use
    :param save_name: What are we evaluating ACC on
    :param writer: Tensorboard logger
    :return:
    """

    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    for i, f_name in enumerate(eval_funcs):

        acc_f = EVAL_FUNCS[f_name]
        all_acc, old_acc, new_acc = acc_f(y_true, y_pred, mask)
        log_name = f'{save_name}_{f_name}'

        if writer is not None:
            writer.add_scalars(log_name,
                               {'Old': old_acc, 'New': new_acc,
                                'All': all_acc}, T)

        if i == 0:
            to_return = (all_acc, old_acc, new_acc)

        if print_output:
            print_str = f'Epoch {T}, {log_name}: All {all_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}'
            print(print_str)

    return to_return


def test_agglo(epoch, feats, targets, mask, save_name, args):
    best_acc = 0
    mask = mask.astype(bool)
    feats = np.concatenate(feats)
    linked = linkage(feats, method="ward")

    gt_dist = linked[:, 2][-args.num_labeled_classes-args.num_unlabeled_classes]
    preds = fcluster(linked, t=gt_dist, criterion='distance')
    test_all_acc_test, test_old_acc_test, test_new_acc_test = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=0, eval_funcs=args.eval_funcs, save_name=save_name)

    dist = linked[:, 2][:-args.num_labeled_classes]
    tolerance = 0
    for d in reversed(dist):
        preds = fcluster(linked, t=d, criterion='distance')
        k = max(preds)
        all_acc_test, old_acc_test, new_acc_test = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                        T=0, eval_funcs=args.eval_funcs, save_name=save_name)

        if old_acc_test > best_acc:                                     # save best labeled acc without knowing GT K
            best_acc = old_acc_test
            best_acc_k = k
            best_acc_d = d
            tolerance = 0
        else:
            tolerance += 1
        
        if tolerance == 50 :
            break
    return test_all_acc_test, test_old_acc_test, test_new_acc_test, best_acc, best_acc_k

