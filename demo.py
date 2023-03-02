import cv2

import autocrop



def run_example(in_path: str, out_path: str):
    """Autocrops the specified image and saves it to disk

    Args:
        in_path (str): Path to image to crop
        out_path (str): Path/filename for image output
    """
    # Load in the image from disk
    image = cv2.imread(in_path)

    if image is None:
        print(f'Could not load {in_path} from disk')
        return

    # Find the product(s) in the image. This return a boundary box
    item_rect = autocrop.find_item_boundaries(image, margin=0.2)

    if item_rect is None:
        print('No distinct items found in image')
        return

    print('Item(s) found in this rectangle:', item_rect)
    left, top, right, bottom = item_rect

    cropped = image[top:bottom, left:right]

    cv2.imwrite(out_path, cropped)


if __name__ == '__main__':
    run_example('example-padded.jpg', 'example-out.jpg')
