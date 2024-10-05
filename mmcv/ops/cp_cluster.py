from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from mmengine.utils import deprecated_api_warning
from torch import Tensor

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['cp_cluster'])


class CPClusterop(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, boxes: Tensor, scores: Tensor, iou_threshold: float,
                sigma: float, min_score: float, method: int,
                offset: int) -> Tuple[Tensor, Tensor]:
        dets = boxes.new_empty((boxes.size(0), 5), device='cpu')
        inds = ext_module.cp_cluster(
            boxes.cpu(),
            scores.cpu(),
            dets.cpu(),
            iou_threshold=float(iou_threshold),
            sigma=float(sigma),
            min_score=float(min_score),
            method=int(method),
            offset=int(offset))
        return dets, inds


array_like_type = Union[Tensor, np.ndarray]


@deprecated_api_warning({'iou_thr': 'iou_threshold'})
def cp_cluster(boxes: array_like_type,
             scores: array_like_type,
             iou_threshold: float = 0.3,
             sigma: float = 0.5,
             min_score: float = 1e-3,
             method: str = 'linear',
             offset: int = 0) -> Tuple[array_like_type, array_like_type]:
    """Dispatch to only CPU Soft NMS implementations.

    The input can be either a torch tensor or numpy array.
    The returned type will always be the same as inputs.

    Args:
        boxes (torch.Tensor or np.ndarray): boxes in shape (N, 4).
        scores (torch.Tensor or np.ndarray): scores in shape (N, ).
        iou_threshold (float): IoU threshold for NMS.
        sigma (float): hyperparameter for gaussian method
        min_score (float): score filter threshold
        method (str): either 'linear' or 'gaussian'
        offset (int, 0 or 1): boxes' width or height is (x2 - x1 + offset).

    Returns:
        tuple: kept dets (boxes and scores) and indice, which always have
        the same data type as the input.

    Example:
        >>> boxes = np.array([[4., 3., 5., 3.],
        >>>                   [4., 3., 5., 4.],
        >>>                   [3., 1., 3., 1.],
        >>>                   [3., 1., 3., 1.],
        >>>                   [3., 1., 3., 1.],
        >>>                   [3., 1., 3., 1.]], dtype=np.float32)
        >>> scores = np.array([0.9, 0.9, 0.5, 0.5, 0.4, 0.0], dtype=np.float32)
        >>> iou_threshold = 0.6
        >>> dets, inds = soft_nms(boxes, scores, iou_threshold, sigma=0.5)
        >>> assert len(inds) == len(dets) == 5
    """

    assert isinstance(boxes, (Tensor, np.ndarray))
    assert isinstance(scores, (Tensor, np.ndarray))
    is_numpy = False
    if isinstance(boxes, np.ndarray):
        is_numpy = True
        boxes = torch.from_numpy(boxes)
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    assert boxes.size(1) == 4
    assert boxes.size(0) == scores.size(0)
    assert offset in (0, 1)
    method_dict = {'naive': 0, 'linear': 1, 'gaussian': 2}
    assert method in method_dict.keys()

    if torch.__version__ == 'parrots':
        dets = boxes.new_empty((boxes.size(0), 5), device='cpu')
        indata_list = [boxes.cpu(), scores.cpu(), dets.cpu()]
        indata_dict = {
            'iou_threshold': float(iou_threshold),
            'sigma': float(sigma),
            'min_score': min_score,
            'method': method_dict[method],
            'offset': int(offset)
        }
        inds = ext_module.cp_cluster(*indata_list, **indata_dict)
    else:
        dets, inds = CPClusterop.apply(boxes.cpu(), scores.cpu(),
                                     float(iou_threshold), float(sigma),
                                     float(min_score), method_dict[method],
                                     int(offset))

    dets = dets[:inds.size(0)]

    if is_numpy:
        dets = dets.cpu().numpy()
        inds = inds.cpu().numpy()
        return dets, inds
    else:
        return dets.to(device=boxes.device), inds.to(device=boxes.device)
