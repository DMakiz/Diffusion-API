from controlnet_aux import (
    CannyDetector,
    ContentShuffleDetector,
    HEDdetector,
    LineartAnimeDetector,
    LineartDetector,
    MediapipeFaceDetector,
    MidasDetector,
    MLSDdetector,
    NormalBaeDetector,
    OpenposeDetector,
    PidiNetDetector,
    SamDetector,
   
)

import numpy as np
import cv2

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y
        
def safer_memory(x):
    return np.ascontiguousarray(x.copy()).copy()


def resize_image_with_pad(input_image, resolution, skip_hwc3=False):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
        
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=interpolation)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode='edge')

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target])

    return safer_memory(img_padded), remove_pad


def scribble_xdog(img, res=512, thr_a=32, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    g1 = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.5)
    g2 = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 5.0)
    dog = (255 - np.min(g2 - g1, axis=2)).clip(0, 255).astype(np.uint8)
    result = np.zeros_like(img, dtype=np.uint8)
    result[2 * (255 - dog) > thr_a] = 255
    return remove_pad(result), True

def none_preprocces(image_path:str):
    return Image.open(image_path)

PREPROCCES_DICT = {
    "Hed": HEDdetector.from_pretrained("lllyasviel/Annotators"),
    "Midas": MidasDetector.from_pretrained("lllyasviel/Annotators"),
    "MLSD": MLSDdetector.from_pretrained("lllyasviel/Annotators"),
    "Openpose": OpenposeDetector.from_pretrained("lllyasviel/Annotators"),
    "PidiNet": PidiNetDetector.from_pretrained("lllyasviel/Annotators"),
    "NormalBae": NormalBaeDetector.from_pretrained("lllyasviel/Annotators"),
    "Lineart": LineartDetector.from_pretrained("lllyasviel/Annotators"),
    "LineartAnime": LineartAnimeDetector.from_pretrained(
        "lllyasviel/Annotators"
    ),
    
    "Canny": CannyDetector(),
    "ContentShuffle": ContentShuffleDetector(),
    "MediapipeFace": MediapipeFaceDetector(),
    "ScribbleXDOG": scribble_xdog,
    "None": none_preprocces
}
    
