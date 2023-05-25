import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

content_path = "Dataset/trainB/"
style_path = "Dataset/trainA/"


def load_data():
    # content process
    content_files = [fl for fl in os.listdir(content_path) if fl.endswith('jpg')]
    content_array = np.zeros((len(content_files), 256, 256, 3), dtype=np.uint8)
    for idx, fl in enumerate(content_files):
        content_array[idx]=Image.open(f"{content_path}{fl}")
    #style processing
    style_files = [fl for fl in os.listdir(style_path) if fl.endswith('jpg')]
    style_array = np.zeros((len(style_files), 256, 256, 3), dtype=np.uint8)
    for idx, fl in enumerate(style_files):
        style_array[idx]=Image.open(f"{style_path}{fl}")
    return content_array, style_array
