import numpy as np

def metrics_update(cur_max_iou, cur_precisions, cur_recalls, iou, precision, recall):
    iou.append(np.mean(cur_max_iou) if len(cur_max_iou) > 0 else 0)
    precision.append(np.mean(cur_precisions) if len(cur_max_iou) > 0 else 0)
    recall.append(np.mean(cur_recalls) if len(cur_max_iou) > 0 else 0)


def find_best_metrics(cur_max_iou, cur_precisions, cur_recalls, i, j, predicts, targets):
    best_iou = 0
    best_precision = 0
    best_recall = 0
    for mask2 in predicts[i]:
        best_iou, best_precision, best_recall = compare_metrics(targets, best_iou, best_precision, best_recall, i, j, mask2)
    cur_iou, cur_precision, cur_recall = best_iou, best_precision, best_recall
    cur_max_iou.append(cur_iou)
    cur_precisions.append(cur_precision)
    cur_recalls.append(cur_recall)
    return cur_max_iou, cur_precisions, cur_recalls


def compare_metrics(all_targets, best_iou, best_precision, best_recall, i, j, mask2):
    mask1 = all_targets[i][j].astype(int)
    mask2 = mask2.astype(int)
    mask1_area = np.count_nonzero(mask1 == 1)
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero(np.logical_and(mask1 == 1, mask2 == 1))
    TP = intersection
    FP = mask2_area - intersection
    FN = mask1_area - intersection
    iou = TP / (TP + FP + FN)
    if TP + FP > 0:
        precision = TP / (TP + FP)
    else:
        precision = 0
    if TP + FN > 0:
        recall = TP / (TP + FN)
    else:
        recall = 0
    if best_iou < iou:
        best_iou = iou
        best_precision = precision
        best_recall = recall
    return best_iou, best_precision, best_recall


def print_metrics(all_predicts, iou, precision, recall):
    iou_50 = (np.array(iou) > 0.5).sum() / len(all_predicts)
    iou_75 = (np.array(iou) > 0.75).sum() / len(all_predicts)
    iou_90 = (np.array(iou) > 0.9).sum() / len(all_predicts)
    iou = np.mean(iou)
    precision = np.mean(precision)
    recall = np.mean(recall)
    print(f'Precision: {precision:.5f}')
    print(f'Recall: {recall:.5f}')
    print(f'IoU: {iou:.5f}')
    print(f'IoU>0.5: {iou_50:.5f}')
    print(f'IoU>0.75: {iou_75:.5f}')
    print(f'IoU>0.9: {iou_90:.5f}')

