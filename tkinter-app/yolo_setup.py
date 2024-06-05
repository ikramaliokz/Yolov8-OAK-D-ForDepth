import cv2
import depthai as dai
import numpy as np
import time


text_color = (0, 0, 255)
bbox_color = (0, 255, 0)
nnBlobPath = "models/yolov8n_openvino_2022.1_6shave.blob"

LABEL_MAP_YOLO = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "TV", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def setup_pipeline_yolo():
    syncNN = True

    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)

    yolo_spatial_det_nn = pipeline.createYoloSpatialDetectionNetwork()
    yolo_spatial_det_nn.setConfidenceThreshold(0.5)
    yolo_spatial_det_nn.setBlobPath(nnBlobPath)
    yolo_spatial_det_nn.setNumClasses(80)  # Adjust based on your model
    yolo_spatial_det_nn.setCoordinateSize(4)
    yolo_spatial_det_nn.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])  # Adjust based on your model
    yolo_spatial_det_nn.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})  # Adjust based on your model
    yolo_spatial_det_nn.setIouThreshold(0.5)
    yolo_spatial_det_nn.setDepthLowerThreshold(100)
    yolo_spatial_det_nn.setDepthUpperThreshold(5000)


    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)

    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutNN = pipeline.create(dai.node.XLinkOut)
    xoutDepth = pipeline.create(dai.node.XLinkOut)

    xoutRgb.setStreamName("rgb")
    xoutNN.setStreamName("detections")
    xoutDepth.setStreamName("depth")

    # Properties
    camRgb.setPreviewSize(640, 640)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setCamera("left")
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setCamera("right")

    # Setting node configs
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    # Align depth map to the perspective of RGB camera, on which inference is done
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setSubpixel(True)
    stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())


    # Linking
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    camRgb.preview.link(yolo_spatial_det_nn.input)
    if syncNN:
        yolo_spatial_det_nn.passthrough.link(xoutRgb.input)
    else:
        camRgb.preview.link(xoutRgb.input)

    yolo_spatial_det_nn.out.link(xoutNN.input)

    stereo.depth.link(yolo_spatial_det_nn.inputDepth)
    yolo_spatial_det_nn.passthroughDepth.link(xoutDepth.input)

    return pipeline