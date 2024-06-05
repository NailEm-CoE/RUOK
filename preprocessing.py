import torch.nn as nn

import numpy as np
import cv2
from PIL import Image

import onnxruntime as ort

from torchvision.transforms.functional import center_crop


class ruok_preprocess(nn.Module):
    """
    This class preprocesses chest X-ray images for RUOK.

    Attributes:
        thorax_model (UnetOnnx): The model used for segmenting thorax in the image.
        center_crop_ratio (float): The ratio for center cropping the image.
        clipLimit (int): The clip limit for CLAHE.
        tileGridSize (tuple): The tile grid size for CLAHE.
        clahe (CLAHE): The CLAHE object for histogram equalization.
    """
    def __init__(self):
        super().__init__()
        self.thorax_model = UnetOnnx()

        self.center_crop_ratio = 0.1
        self.clipLimit = 1
        self.tileGridSize = (4, 4)

        self.clahe = cv2.createCLAHE(self.clipLimit, self.tileGridSize)

    def forward(self, image: Image):
        """
        Preprocesses the input image.

        Args:
            image (Image): The input image to be preprocessed.

        Returns:
            Image: The preprocessed image.
        """
        # get org size
        org_height, org_width = image.size
        # convert image to RGB to confirm consistancy
        image = image.convert('RGB')

        # get numpy image
        np_image = np.asarray(image)

        # preprocess image for thorax model
        preprocessed_image_for_thorax_model = thorax_model_preprocess(np_image)
        batched_preprocessed_image_for_thorax_model = preprocessed_image_for_thorax_model[np.newaxis, :, :, np.newaxis]
        # get thorax mask
        thorax_mask = self.thorax_model(batched_preprocessed_image_for_thorax_model)

        # cast thorax mask to uint8 image
        thorax_mask = np.uint8(thorax_mask * 255.)
        # resize thorax mask to match original image
        resized_thorax_mask = cv2.resize(thorax_mask, (org_width, org_height))

        # upright chest x-ray image
        thorax_image, bbox, M = uprightChestCxr(image=np_image, mask=resized_thorax_mask)

        # cast thorax_image to PIL Image
        thorax_image = Image.fromarray(thorax_image)
        # get size for center crop
        output_size = np.asarray(thorax_image.size)
        # perform center crop
        cropped_thorax_image = center_crop(thorax_image, (output_size - (output_size * self.center_crop_ratio)).astype(int))

        # resize to 224x224
        resized_cropped_thorax_image = cv2.resize(np.asarray(cropped_thorax_image), (224, 224), cv2.INTER_CUBIC)

        # apply clahe
        resized_cropped_thorax_image = cv2.cvtColor(resized_cropped_thorax_image, cv2.COLOR_BGR2GRAY)
        clahe_resized_cropped_thorax_image = self.clahe.apply(resized_cropped_thorax_image)

        # cast to PIL Image
        clahe_resized_cropped_thorax_image = Image.fromarray(clahe_resized_cropped_thorax_image)

        return clahe_resized_cropped_thorax_image.convert('RGB')


def thorax_model_preprocess(image):
    """
    Preprocesses the input image for the thorax model.

    Args:
        image (numpy array): The input image.

    Returns:
        numpy array: The preprocessed image.
    """
    # make image grayscale
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Histogram equalization on input image
    image = cv2.equalizeHist(image)

    # resize to 256x256 using inter area interpolation
    image_resize = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)

    # scale image to 0..1 range
    image_scaled = (image_resize / 255.0).astype('float32')

    return image_scaled


class UnetOnnx:
    """
    This class handles the inference for the thorax segmentation model.
    
    Attributes:
        THORAX_SEGMENT_MODEL_INPUT_NAME (str): The input name for the thorax model.
        thorax_SEGMENT_SESS (InferenceSession): The ONNX runtime inference session for the thorax model.
    """
    def __init__(self):
        self.THORAX_SEGMENT_MODEL_INPUT_NAME = "input_1"
        self.thorax_SEGMENT_SESS = ort.InferenceSession(
            "weights/unet_thorax_cropping_model.onnx",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )


    def __call__(self, image, threshold=0.5):
        """
        Runs the thorax model to get the segmentation mask.

        Args:
            image (numpy array): The input image.
            threshold (float): The threshold for the binary mask.

        Returns:
            numpy array: The binary thorax mask.
        """
        # Get thorax mask from model
        thorax_mask = self.thorax_SEGMENT_SESS.run(None, { self.THORAX_SEGMENT_MODEL_INPUT_NAME: image },)
        
        # Apply threshold to the mask
        _, thresholed_thorax_mask = cv2.threshold(thorax_mask[0][0], threshold, 1, cv2.THRESH_BINARY)

        return thresholed_thorax_mask


def uprightChestCxr(image, mask):
    """
    Uprights the chest X-ray image using the thorax mask.

    Args:
        image (numpy array): The input image.
        mask (numpy array): The thorax mask.

    Returns:
        tuple: A tuple containing:
            - numpy array: The uprighted image.
            - tuple: The bounding box (x1, y1, x2, y2) indicating the top-left and bottom-right coordinates of the cropped region.
            - numpy array: The rotation matrix used to upright the image.
    """

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # find largest contours
    largest_contour = max(contours, key=cv2.contourArea)

    # Get all necessary points
    cnt = largest_contour[:, 0, :]
    point_a = cnt[np.argmin(cnt[:, 0] - cnt[:, 1])] # left
    point_b = cnt[np.argmax(cnt[:, 0] + cnt[:, 1])] # right
    point_d = ((point_a + point_b) / 2).astype('int') # middle
    point_c = findCentroidFromContour(cnt) # centroid

    # Calculate vertical rotation
    slope_v = (point_c[1] - point_d[1]) / (point_c[0] - point_d[0])
    rotate_v = 90 - abs(np.arctan(slope_v) * (180 / np.pi))

    # Calculate horizontal rotation
    slope_h = (point_a[1] - point_b[1]) / (point_a[0] - point_b[0])
    rotate_h = np.arctan(slope_h) * (180 / np.pi)

    # Determine the angle to rotate
    if np.abs(rotate_v) < np.abs(rotate_h):
        angle = rotate_v * -1 if slope_v > 0 else rotate_v
    else:
        angle = rotate_h

    # Rotate image and matrix
    rotated_image, M = rotate_bound(image, angle, point_c)

    # Apply rotation matrix to contours
    rcnt = (np.matmul(M[:, 0:2], cnt.T).T + M[:, 2]).astype('int')
    (x, y, w, h) = cv2.boundingRect(rcnt)

    # Define borders for cropping
    borders = np.asarray((0.025 * h, 0.025 * w, 0.05 * h, 0.025 * w), dtype='int') # top, right, bottom, left
    st_crop = (max(x - borders[3], 0), max(y - borders[0], 0))
    en_crop = (min(x + w + borders[1], image.shape[1]), min(y + h + borders[2], image.shape[0]))

    # Crop the rotated image
    cropped_rotated_image = rotated_image[st_crop[1]:en_crop[1], st_crop[0]:en_crop[0]]
    
    # Bounding box with x1, y1, x2, y2
    bbox = (st_crop[0], st_crop[1], en_crop[0], en_crop[1])

    return cropped_rotated_image, bbox, M


def findCentroidFromContour(c):
    """
    Finds the centroid of a contour.

    Args:
        c (numpy array): The contour.

    Returns:
        numpy array: The coordinates of the centroid.
    """

    # Calculate moments of the contour
    M = cv2.moments(c)

    # Ensure valid contour area
    if M["m00"] <= 0 :
        return False

    # Calculate centroid coordinates
    cX = M["m10"] / M["m00"]
    cY = M["m01"] / M["m00"]

    return np.asarray((int(cX), int(cY)))


def rotate_bound(image, angle, center=None):
    """
    Rotates the image around the specified center.

    Args:
        image (numpy array): The input image.
        angle (float): The angle of rotation.
        center (tuple): The center of rotation.

    Returns:
        tuple: The rotated image and the rotation matrix.
    """
    (h, w) = image.shape[:2]
    if center is None:
        (cX, cY) = (w // 2, h // 2)
    else:
        (cX, cY) = center

    # Get rotation matrix
    M = cv2.getRotationMatrix2D((int(cX), int(cY)), angle, 1.0) # cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, M, (w, h)), M
