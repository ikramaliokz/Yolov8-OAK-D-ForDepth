import cv2
import depthai as dai
import blobconverter
import numpy as np

# Define frame size and model input size
FRAME_SIZE = (640, 400)
DET_INPUT_SIZE = (640, 640)
model_name = None # "yolo-v4-tiny-tf"
zoo_type = "depthai"
blob_path = 'models/yolov8n_openvino_2022.1_6shave.blob'

# Create pipeline
pipeline = dai.Pipeline()

# Define source - RGB camera
cam = pipeline.createColorCamera()
cam.setPreviewSize(FRAME_SIZE[0], FRAME_SIZE[1])
cam.setInterleaved(False)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# Define mono camera sources for stereo depth
mono_left = pipeline.createMonoCamera()
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
mono_right = pipeline.createMonoCamera()
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Create stereo depth node
stereo = pipeline.createStereoDepth()
stereo.setLeftRightCheck(True)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)  # Align depth map to RGB camera

# Linking
mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

cam.setBoardSocket(dai.CameraBoardSocket.RGB)

# Convert model from OMZ to blob
if model_name is not None:
    blob_path = blobconverter.from_zoo(
        name=model_name,
        shaves=6,
        zoo_type=zoo_type
    )

# Define YOLO detection NN node
yolo_spatial_det_nn = pipeline.createYoloSpatialDetectionNetwork()
yolo_spatial_det_nn.setConfidenceThreshold(0.5)
yolo_spatial_det_nn.setBlobPath(blob_path)
yolo_spatial_det_nn.setNumClasses(80)  # Adjust based on your model
yolo_spatial_det_nn.setCoordinateSize(4)
yolo_spatial_det_nn.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])  # Adjust based on your model
yolo_spatial_det_nn.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})  # Adjust based on your model
yolo_spatial_det_nn.setIouThreshold(0.5)
yolo_spatial_det_nn.setDepthLowerThreshold(100)
yolo_spatial_det_nn.setDepthUpperThreshold(5000)

# Define detection input config
det_manip = pipeline.createImageManip()
det_manip.initialConfig.setResize(DET_INPUT_SIZE[0], DET_INPUT_SIZE[1])
det_manip.initialConfig.setKeepAspectRatio(False)
det_manip.setMaxOutputFrameSize(1228800)
# Linking
cam.preview.link(det_manip.inputImage)
det_manip.out.link(yolo_spatial_det_nn.input)
stereo.depth.link(yolo_spatial_det_nn.inputDepth)

# Create preview output
x_preview_out = pipeline.createXLinkOut()
x_preview_out.setStreamName("preview")
cam.preview.link(x_preview_out.input)

# Create detection output
det_out = pipeline.createXLinkOut()
det_out.setStreamName('det_out')
yolo_spatial_det_nn.out.link(det_out.input)

# Create disparity output
disparity_out = pipeline.createXLinkOut()
disparity_out.setStreamName("disparity")
stereo.disparity.link(disparity_out.input)

import time
# Create and start the pipeline
with dai.Device(pipeline) as device:
    preview_queue = device.getOutputQueue(name="preview", maxSize=4, blocking=False)
    det_queue = device.getOutputQueue(name="det_out", maxSize=4, blocking=False)
    disparity_queue = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
    # num_frames = 100
    # start_time = time.time()
    # for _ in range(num_frames):
    #     preview_queue.get()
    # end_time = time.time()
    # camera_fps = num_frames / (end_time - start_time)
    # print(f"Measured Camera FPS: {camera_fps}")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 12, (FRAME_SIZE[0], FRAME_SIZE[1]))

    while True:
        in_preview = preview_queue.get()
        in_det = det_queue.get()
        in_disparity = disparity_queue.get()

        frame = in_preview.getCvFrame()
        detections = in_det.detections
        disparity_frame = in_disparity.getFrame()
        disparity_frame = (disparity_frame * (255 / stereo.initialConfig.getMaxDisparity())).astype(np.uint8)

        # Visualize detections
        for detection in detections:
            # Denormalize bounding box
            x1 = int(detection.xmin * FRAME_SIZE[0])
            y1 = int(detection.ymin * FRAME_SIZE[1])
            x2 = int(detection.xmax * FRAME_SIZE[0])
            y2 = int(detection.ymax * FRAME_SIZE[1])
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Put depth information
            depth = int(detection.spatialCoordinates.z)
            label = f"Depth: {depth} mm"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame into the file 'output.avi'
        out.write(frame)

        # Display frames
        cv2.imshow("Preview", frame)
        disparity_frame = cv2.resize(disparity_frame, FRAME_SIZE)
        cv2.imshow("Disparity", disparity_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()