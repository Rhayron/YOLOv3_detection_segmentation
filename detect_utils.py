"""
Run inference on images, videos, directories, streams, etc.
Usage:
    $ python path/to/detect.py --weights yolov3.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

'''
This function loads the weights of a YOLOv3 model and uses it to detect objects
in the input images. More than one image can be passed at once to reduce the time
impact from loading the model multiple times.

input:
    weights    (str)        : path to the .pt weights file. Default is the 'best.pt' in the current dir.
    source     (str)        : path to either the directory that has the images, or a single image.
    imgsz      (int) : size of the YOLO window for detection. Default is 640.
    conf_thres (float)      : detections with less confidence than this will be ommited. Default is 0.25
    iou_thres  (float)      : IOU threshold used during NMS to determine if boxes are not the same object. Default is 0.45
    max_det    (int)        : maximum detections per image. Default is 1000.
    device     (str)        : cuda device (0, 1, 2, 3 or cpu). If default is empty, it will use be the gpu if pytorch was configured for that.
    classes    (list of int): the indexes of the classes that will be detected ie [0, 1, 14]. Without this input, it detects all classes.
output:
    list of dicts in the format, per item 
        img_name (str)                   : path for that particular image, including filename and extension.
        boxes    (list of list of floats): a list with all detections for that image in the format [x, y, w, h, confidence, class index] per item.
'''
@torch.no_grad()
def detect_boxes(    weights='best_tcc.pt',  # model.pt path(s)
            source='./sample_images/',  # file/dir/URL/glob
            imgsz=640,  # inference size (pixels)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            ):
    source = str(source)
    imgsz = [imgsz, imgsz]
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    detections = []
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        # not using agnostic_nms
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, False, max_det=max_det)
        dt[2] += time_sync() - t3
        box_list = {}
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() # for save_crop
            box_list["img_name"] = path
            boxes = []
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4),) / gn).view(-1).tolist()  # normalized xywh              
                    xywh.append(conf.item())
                    xywh.append(cls.item())
                    boxes.append(xywh)
            box_list["boxes"] = boxes
        detections.append(box_list)
    
    t = tuple(x / seen * 1E3 for x in dt)  # speed per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    
    return detections