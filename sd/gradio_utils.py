
import copy
import math
import os
import urllib.request
from typing import List, Optional, Tuple

import numpy as np
import PIL
import PIL.Image
import PIL.ImageDraw
import torch
import torch.optim
from tqdm import tqdm
import ipdb

def tensor_to_PIL(img: torch.Tensor) -> PIL.Image.Image:
    """
    Converts a tensor image to a PIL Image.

    Args:
        img (torch.Tensor): The tensor image of shape [batch_size, num_channels, height, width].

    Returns:
        A PIL Image object.
    """
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return PIL.Image.fromarray(img[0].cpu().numpy(), "RGB")


def get_ellipse_coords(
    point: Tuple[int, int], radius: int = 5
) -> Tuple[int, int, int, int]:
    """
    Returns the coordinates of an ellipse centered at the given point.

    Args:
        point (Tuple[int, int]): The center point of the ellipse.
        radius (int): The radius of the ellipse.

    Returns:
        A tuple containing the coordinates of the ellipse in the format (x_min, y_min, x_max, y_max).
    """
    center = point
    return (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )



def draw_handle_target_points(
        img: PIL.Image.Image,
        # handle_points: List[Tuple[int, int]],
        target_points: List[Tuple[int, int]],
        radius: int = 5):
    """
    Draws handle and target points with arrow pointing towards the target point.

    Args:
        img (PIL.Image.Image): The image to draw on.
        handle_points (List[Tuple[int, int]]): A list of handle [x,y] points.
        target_points (List[Tuple[int, int]]): A list of target [x,y] points.
        radius (int): The radius of the handle and target points.
    """
    if not isinstance(img, PIL.Image.Image):
        img = PIL.Image.fromarray(img)

    # if len(handle_points) == len(target_points) + 1:
    #     target_points = copy.deepcopy(target_points) + [None]

    draw = PIL.ImageDraw.Draw(img)
    for handle_point, target_point in zip(target_points, target_points):
        # handle_point = [handle_point[1], handle_point[0]]
        # Draw the handle point
        # ipdb.set_trace()

        target_coords = get_ellipse_coords(target_point, radius)
        draw.ellipse((target_coords), fill="red")
        
    return np.array(img)


