# Example Code: Autocrop product images using OpenCV
This is an example sampling of code that I have implemented in the past to automatically crop excess padding from product images.

It uses edge detection (contour) to find the most prominent objects in the image, filtering noise and small items. Once all of the items in the image are found, you can crop the image down to a more constrained padding.

## Installation:

### OS Package requirements:
- Linux: `sudo apt-get install python-opencv`
- Mac: TODO - OS specific instructions
- Windows: `pip install opencv-python`

### Install python dependencies
```sh
pip install venv
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

### Run demo:
In the terminal, run "`python demo.py`".

This reads in `example-padded.jpg` and creates a cropped version in `example-out.jpg`.

<br />


<sub><sup>
Example image source: https://expertphotography.com/budget-product-photography/
<sup></sub>