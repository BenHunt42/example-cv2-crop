from typing import List

import cv2


# Theres probably a better way to define this
Rect = List[int]  # [int, int, int, int]


def find_item_boundaries(src, margin=0.04) -> Rect:
    """Autocrap an image around items not touching the edges of the canvas

    Args:
        src (numpy.array): Image converted to a numpy array
        margin (float, optional): How much space to add around the found item. Defaults to 0.04.

    Returns:
        Rect: A four item list representing the bounding box of the found items:
            left (int): Distance from the left of the image to the left edge of found item
            top (int): Distance from the top of the image to the top edge found item
            right (int): Distance from the left of the image to the right edge of found item
            bottom (int): Distance from the top of the image to the bottom edge found item
    """
    # Clamp the image at a max of 500x500 for faster processing
    src_gray = shrink(src, 500, 500)
    scale_w_ratio = src.shape[1] / src_gray.shape[1]
    scale_h_ratio = src.shape[0] / src_gray.shape[0]

    # Convert to grayscale and blur
    src_gray = cv2.cvtColor(src_gray, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.blur(src_gray, (3, 3))

    # Edge detect
    threshold = 50
    canny_output = cv2.Canny(src_gray, threshold, threshold * 3)

    # Get coords for those edges
    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_poly = [None]*len(contours)
    bound_rect = [None]*len(contours)

    # Find the bounding rects for each contour
    for i, contour in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(contour, 3, True)
        bound_rect[i] = cv2.boundingRect(contours_poly[i])

    crop_rect = None

    for i in range(len(contours)):
        # Ignore stuff touching edges
        if is_contour_on_edge(contours[i], src_gray.shape[1], src_gray.shape[0]):
            continue

        # Determine a rectangle that encompasses all found items
        if crop_rect is None:
            # Set to match the bounding rect if not set yet
            crop_rect = [
                bound_rect[i][0],
                bound_rect[i][1],
                bound_rect[i][0] + bound_rect[i][2],
                bound_rect[i][1] + bound_rect[i][3],
            ]
        else:
            crop_rect = [
                min(crop_rect[0], bound_rect[i][0]),
                min(crop_rect[1], bound_rect[i][1]),
                max(crop_rect[2], bound_rect[i][0] + bound_rect[i][2]),
                max(crop_rect[3], bound_rect[i][1] + bound_rect[i][3]),
            ]

    # No shapes detected
    if crop_rect is None:
        return None

    crop_rect = scale_rect_coords(crop_rect, scale_w_ratio, scale_h_ratio)
    w_padd = int((crop_rect[2] - crop_rect[0]) * margin)
    h_padd = int((crop_rect[3] - crop_rect[1]) * margin)
    crop_rect = pad_rect_coords(crop_rect, w_padd, h_padd, src.shape)

    return crop_rect


def scale_rect_coords(rect: Rect, w_ratio, h_ratio) -> Rect:
    """Utility function that scales a given rectangle

    Args:
        rect (Rect): List of coords representing a box:

        w_ratio (float): Amount to scale the x dimension
        h_ratio (float): Amount to scale the y dimension

    Returns:
        Rect: Scaled rectangle
    """
    return [
        int(rect[0] * w_ratio),
        int(rect[1] * h_ratio),
        int(rect[2] * w_ratio),
        int(rect[3] * h_ratio),
    ]


def pad_rect_coords(rect: Rect, padding_w: int, padding_h: int, max_size: List[int]) -> Rect:
    """Adds a fixed padding to the given rectangle without overflowing a fixed size

    Args:
        rect (Rect): The rectangle to pad
        padding_w (int): Amount of padding to add to the x dimension
        padding_h (int): Amount of padding to add to the y dimension
        max_size ([int, int]): Two item list containing max_width and max_height

    Returns:
        Rect: The padded rectangle
    """
    return [
        max(0, rect[0] - padding_w),
        max(0, rect[1] - padding_h),
        min(max_size[1], rect[2] + padding_w),
        min(max_size[0], rect[3] + padding_h),
    ]


def is_contour_on_edge(contour: list, width: int, height: int) -> bool:
    """Determine whether or not the provided contour is on the edge of the
    image bounds.

    Args:
        contour (list): A nested list of coords that represent a contour
        width (int): The width of the image
        height (int): The height of the image

    Returns:
        bool: True if the contour is on the edge. False otherwise.
    """
    left_edge = width * 0.01
    right_edge = width - left_edge
    top_edge = height * 0.01
    bottom_edge = height - top_edge

    for group in contour:
        for point in group:
            if point[0] < left_edge or point[0] > right_edge:
                return True

            if point[1] < top_edge or point[1] > bottom_edge:
                return True
    return False


def shrink(image, max_width: int, max_height: int):
    """Ensures an image is no larger than the given max width/height

    Args:
        image (numpy.array): The image to shrink
        max_width (int): The max width of the resulting image
        max_height (int): The max height of the resulting image

    Raises:
        Exception: If the image could not be resized

    Returns:
        numpy.array: The source image if it was already small enough.
            Otherwise, a smaller version of the source image.
    """
    image_width = image.shape[1]
    image_height = image.shape[0]

    # Dont attempt resize if already small enough
    if image_width <= max_width and image_height <= max_height:
        return image

    dest_width = max_width
    dest_height = max_height
    ratio = image_width / image_height

    # Width is larger than height
    if image_width >= image_height:
        dest_height = int(max_height / ratio)
    else:
        dest_width = int(max_width / ratio)

    image = cv2.resize(image, (dest_width, dest_height))

    if image is None:
        raise Exception('Could not shrink image')

    return image
