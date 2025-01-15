import numpy as np
from src import box_process
import torch

def post_process_yolov10(input_data, classe, obj_thresh):
    max_det, nc = 300, len(classe)

    boxes, scores = [], []
    default_branch = 3
    pair_per_branch = len(input_data) // default_branch

    for i in range(default_branch):
        boxes.append(box_process.box_process(input_data[pair_per_branch*i]))
        scores.append(input_data[pair_per_branch*i+1])

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)
        
    boxes = [sp_flatten(_v) for _v in boxes]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = torch.from_numpy(np.expand_dims(np.concatenate(boxes), axis=0))
    scores = torch.from_numpy(np.expand_dims(np.concatenate(scores), axis=0))

    max_scores = scores.amax(dim=-1)
    max_scores, index = torch.topk(max_scores, max_det, axis=-1)
    index = index.unsqueeze(-1)
    boxes = torch.gather(boxes, dim=1, index=index.repeat(1, 1, boxes.shape[-1]))
    scores = torch.gather(scores, dim=1, index=index.repeat(1, 1, scores.shape[-1]))

    scores, index = torch.topk(scores.flatten(1), max_det, axis=-1)
    labels = index % nc
    index = index // nc
    boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))

    preds = torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)

    mask = preds[..., 4] > obj_thresh

    preds = [p[mask[idx]] for idx, p in enumerate(preds)][0]
    boxes = preds[..., :4].numpy()
    scores =  preds[..., 4].numpy()
    classes = preds[..., 5].numpy().astype(np.int64)

    return boxes, classes, scores
