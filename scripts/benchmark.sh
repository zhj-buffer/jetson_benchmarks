# too many fallback, not recommended run on dla.
sudo nvpmodel -m 0
sudo jetson_clocks
#/usr/src/tensorrt/bin/trtexec ${common} --deploy=vgg19_N2.prototxt --output=prob --int8 --batch=8 --useDLACore=0 --allowGPUFallback --best --workspace=4096  

#[02/19/2022-05:35:26] [I] [TRT] ---------- Layers Running on DLA ---------- 
#[02/19/2022-05:35:26] [I] [TRT] [DlaLayer] {ForeignNode[conv1_1...pool5]}
#[02/19/2022-05:35:26] [I] [TRT] [DlaLayer] {ForeignNode[relu6...fc8]}
#[02/19/2022-05:35:26] [I] [TRT] ---------- Layers Running on GPU ----------
#[02/19/2022-05:35:26] [I] [TRT] [GpuLayer] SHUFFLE: shuffle_between_pool5_and_fc6 
#[02/19/2022-05:35:26] [I] [TRT] [GpuLayer] CONVOLUTION: fc6
#[02/19/2022-05:35:26] [I] [TRT] [GpuLayer] SOFTMAX: prob
avgRuns=1
iterations=1
warmUp=1
useManagedMemory=true

#common="--avgRuns=1  --iterations=1 --warmUp=1 --useManagedMemory --streams=2 --exportTimes="
time=`date "+%Y-%m-%d_%H:%M:%S"`
common="--exportTimes=$time"

echo $common

echo "inception_v4.prototxt on GPU"
/usr/src/tensorrt/bin/trtexec ${common} --deploy=inception_v4.prototxt --output=prob  --best --workspace=4096   --batch=1
/usr/src/tensorrt/bin/trtexec ${common} --deploy=inception_v4.prototxt --output=prob  --best --workspace=4096   --batch=2  
/usr/src/tensorrt/bin/trtexec ${common} --deploy=inception_v4.prototxt --output=prob  --best --workspace=4096   --batch=4 
/usr/src/tensorrt/bin/trtexec ${common} --deploy=inception_v4.prototxt --output=prob  --best --workspace=4096   --batch=8 
/usr/src/tensorrt/bin/trtexec ${common} --deploy=inception_v4.prototxt --output=prob  --best --workspace=4096   --batch=16

echo "vgg19_N2.prototxt on GPU"
/usr/src/tensorrt/bin/trtexec ${common} --deploy=vgg19_N2.prototxt --output=prob  --best --workspace=4096   --batch=1
/usr/src/tensorrt/bin/trtexec ${common} --deploy=vgg19_N2.prototxt --output=prob  --best --workspace=4096   --batch=2  
/usr/src/tensorrt/bin/trtexec ${common} --deploy=vgg19_N2.prototxt --output=prob  --best --workspace=4096   --batch=4 
/usr/src/tensorrt/bin/trtexec ${common} --deploy=vgg19_N2.prototxt --output=prob  --best --workspace=4096   --batch=8 
/usr/src/tensorrt/bin/trtexec ${common} --deploy=vgg19_N2.prototxt --output=prob  --best --workspace=4096   --batch=16

echo "inception_v4.prototxt on DLA"
# only the last layer prob fallback to GPU, accepted as default.
/usr/src/tensorrt/bin/trtexec ${common} --deploy=inception_v4.prototxt --output=prob  --batch=1 --best --workspace=4096 --useDLACore=0 --allowGPUFallback
/usr/src/tensorrt/bin/trtexec ${common} --deploy=inception_v4.prototxt --output=prob  --batch=2 --best --workspace=4096 --useDLACore=0 --allowGPUFallback
/usr/src/tensorrt/bin/trtexec ${common} --deploy=inception_v4.prototxt --output=prob  --batch=4 --best --workspace=4096 --useDLACore=0 --allowGPUFallback
/usr/src/tensorrt/bin/trtexec ${common} --deploy=inception_v4.prototxt --output=prob  --batch=8 --best --workspace=4096 --useDLACore=0 --allowGPUFallback
/usr/src/tensorrt/bin/trtexec ${common} --deploy=inception_v4.prototxt --output=prob  --batch=16 --best --workspace=4096 --useDLACore=0 --allowGPUFallback



# no dla as 
#[02/19/2022-05:33:48] [I] [TRT] ---------- Layers Running on DLA ---------- 
#[02/19/2022-05:33:48] [I] [TRT] [DlaLayer] {ForeignNode[node_of_9...node_of_15]}
#[02/19/2022-05:33:48] [I] [TRT] ---------- Layers Running on GPU ----------
#[02/19/2022-05:33:48] [I] [TRT] [GpuLayer] SHUFFLE: node_of_17 + node_of_18
#[02/19/2022-05:33:48] [I] [TRT] [GpuLayer] SHUFFLE: node_of_output_0
echo "super_resolution_bsd500 on GPU"
/usr/src/tensorrt/bin/trtexec ${common}  --onnx=super_resolution_bsd500-bs1.onnx   --best --workspace=4096
/usr/src/tensorrt/bin/trtexec ${common}  --onnx=super_resolution_bsd500-bs2.onnx   --best --workspace=4096
/usr/src/tensorrt/bin/trtexec ${common}  --onnx=super_resolution_bsd500-bs4.onnx   --best --workspace=4096
/usr/src/tensorrt/bin/trtexec ${common}  --onnx=super_resolution_bsd500-bs8.onnx   --best --workspace=4096


#[02/19/2022-05:42:49] [I] [TRT] ---------- Layers Running on DLA ---------- 
#[02/19/2022-05:42:49] [I] [TRT] [DlaLayer] {ForeignNode[conv2d_1/convolution...(Unnamed Layer* 67) [Deconvolution]]}
#[02/19/2022-05:42:49] [I] [TRT] [DlaLayer] {ForeignNode[conv2d_transpose_1/BiasAdd...(Unnamed Layer* 95) [Deconvolution]]}
#[02/19/2022-05:42:49] [I] [TRT] [DlaLayer] {ForeignNode[conv2d_transpose_2/BiasAdd...(Unnamed Layer* 123) [Deconvolution]]} 
#[02/19/2022-05:42:49] [I] [TRT] [DlaLayer] {ForeignNode[conv2d_transpose_3/BiasAdd...(Unnamed Layer* 151) [Deconvolution]]}
#[02/19/2022-05:42:49] [I] [TRT] [DlaLayer] {ForeignNode[conv2d_transpose_4/BiasAdd...conv2d_19/Sigmoid_HL_1804289383]}
#[02/19/2022-05:42:49] [I] [TRT] ---------- Layers Running on GPU ----------
#[02/19/2022-05:42:49] [I] [TRT] [GpuLayer] SLICE: conv2d_transpose_1/conv2d_transpose
#[02/19/2022-05:42:49] [I] [TRT] [GpuLayer] SLICE: conv2d_transpose_2/conv2d_transpose
#[02/19/2022-05:42:49] [I] [TRT] [GpuLayer] SLICE: conv2d_transpose_3/conv2d_transpose
#[02/19/2022-05:42:49] [I] [TRT] [GpuLayer] SLICE: conv2d_transpose_4/conv2d_transpose
#[02/19/2022-05:42:49] [I] [TRT] [GpuLayer] SHUFFLE: conv2d_19/Sigmoid
echo "unet-segmentation on GPU"
/usr/src/tensorrt/bin/trtexec ${common} --uff=unet-segmentation.uff  --uffInput="input_1,1,512,512" --output=conv2d_19/Sigmoid --batch=1 --best --workspace=4096
/usr/src/tensorrt/bin/trtexec ${common} --uff=unet-segmentation.uff  --uffInput="input_1,1,512,512" --output=conv2d_19/Sigmoid --batch=2 --best --workspace=4096
/usr/src/tensorrt/bin/trtexec ${common} --uff=unet-segmentation.uff  --uffInput="input_1,1,512,512" --output=conv2d_19/Sigmoid --batch=4 --best --workspace=4096
/usr/src/tensorrt/bin/trtexec ${common} --uff=unet-segmentation.uff  --uffInput="input_1,1,512,512" --output=conv2d_19/Sigmoid --batch=8 --best --workspace=4096
/usr/src/tensorrt/bin/trtexec ${common} --uff=unet-segmentation.uff  --uffInput="input_1,1,512,512" --output=conv2d_19/Sigmoid --batch=16 --best --workspace=4096

echo "pose_estimation on GPU"
/usr/src/tensorrt/bin/trtexec ${common} --deploy=pose_estimation.prototxt --output=Mconv7_stage2_L2  --batch=1   --best --workspace=4096
/usr/src/tensorrt/bin/trtexec ${common} --deploy=pose_estimation.prototxt --output=Mconv7_stage2_L2  --batch=2   --best --workspace=4096
/usr/src/tensorrt/bin/trtexec ${common} --deploy=pose_estimation.prototxt --output=Mconv7_stage2_L2  --batch=4   --best --workspace=4096
/usr/src/tensorrt/bin/trtexec ${common} --deploy=pose_estimation.prototxt --output=Mconv7_stage2_L2  --batch=8   --best --workspace=4096
/usr/src/tensorrt/bin/trtexec ${common} --deploy=pose_estimation.prototxt --output=Mconv7_stage2_L2  --batch=16  --best --workspace=4096

echo "pose_estimation on DLA"
/usr/src/tensorrt/bin/trtexec ${common} --deploy=pose_estimation.prototxt --output=Mconv7_stage2_L2  --batch=1 --useDLACore=0  --best --workspace=4096
/usr/src/tensorrt/bin/trtexec ${common} --deploy=pose_estimation.prototxt --output=Mconv7_stage2_L2  --batch=2 --useDLACore=0  --best --workspace=4096
/usr/src/tensorrt/bin/trtexec ${common} --deploy=pose_estimation.prototxt --output=Mconv7_stage2_L2  --batch=4 --useDLACore=0  --best --workspace=4096
/usr/src/tensorrt/bin/trtexec ${common} --deploy=pose_estimation.prototxt --output=Mconv7_stage2_L2  --batch=8 --useDLACore=0  --best --workspace=4096
/usr/src/tensorrt/bin/trtexec ${common} --deploy=pose_estimation.prototxt --output=Mconv7_stage2_L2  --batch=16 --useDLACore=0  --best --workspace=4096

echo "yolov3-tiny-416 on GPU"
/usr/src/tensorrt/bin/trtexec ${common}  --onnx=yolov3-tiny-416-bs1.onnx   --best --workspace=4096
/usr/src/tensorrt/bin/trtexec ${common}  --onnx=yolov3-tiny-416-bs2.onnx   --best --workspace=4096
/usr/src/tensorrt/bin/trtexec ${common}  --onnx=yolov3-tiny-416-bs4.onnx   --best --workspace=4096
/usr/src/tensorrt/bin/trtexec ${common}  --onnx=yolov3-tiny-416-bs8.onnx   --best --workspace=4096
/usr/src/tensorrt/bin/trtexec ${common}  --onnx=yolov3-tiny-416-bs16.onnx   --best --workspace=4096


echo "yolov3-tiny-416 on DLA"
/usr/src/tensorrt/bin/trtexec ${common}  --onnx=yolov3-tiny-416-bs1.onnx   --best --workspace=4096  --useDLACore=0  
/usr/src/tensorrt/bin/trtexec ${common}  --onnx=yolov3-tiny-416-bs2.onnx   --best --workspace=4096  --useDLACore=0
/usr/src/tensorrt/bin/trtexec ${common}  --onnx=yolov3-tiny-416-bs4.onnx   --best --workspace=4096  --useDLACore=0
/usr/src/tensorrt/bin/trtexec ${common}  --onnx=yolov3-tiny-416-bs8.onnx   --best --workspace=4096  --useDLACore=0
/usr/src/tensorrt/bin/trtexec ${common}  --onnx=yolov3-tiny-416-bs16.onnx   --best --workspace=4096 --useDLACore=0


echo "ResNet50_224x224 GPU"
/usr/src/tensorrt/bin/trtexec ${common} --deploy=ResNet50_224x224.prototxt --output=prob  --batch=1   --best --workspace=4096
/usr/src/tensorrt/bin/trtexec ${common} --deploy=ResNet50_224x224.prototxt --output=prob  --batch=2   --best --workspace=4096
/usr/src/tensorrt/bin/trtexec ${common} --deploy=ResNet50_224x224.prototxt --output=prob  --batch=4   --best --workspace=4096
/usr/src/tensorrt/bin/trtexec ${common} --deploy=ResNet50_224x224.prototxt --output=prob  --batch=8   --best --workspace=4096
/usr/src/tensorrt/bin/trtexec ${common} --deploy=ResNet50_224x224.prototxt --output=prob  --batch=16  --best --workspace=4096

echo "ResNet50_224x224 on DLA"
/usr/src/tensorrt/bin/trtexec ${common} --deploy=ResNet50_224x224.prototxt --output=prob  --batch=1 --useDLACore=0  --allowGPUFallback  --best --workspace=4096
/usr/src/tensorrt/bin/trtexec ${common} --deploy=ResNet50_224x224.prototxt --output=prob  --batch=2 --useDLACore=0  --allowGPUFallback  --best --workspace=4096
/usr/src/tensorrt/bin/trtexec ${common} --deploy=ResNet50_224x224.prototxt --output=prob  --batch=4 --useDLACore=0  --allowGPUFallback  --best --workspace=4096
/usr/src/tensorrt/bin/trtexec ${common} --deploy=ResNet50_224x224.prototxt --output=prob  --batch=8 --useDLACore=0  --allowGPUFallback  --best --workspace=4096
/usr/src/tensorrt/bin/trtexec ${common} --deploy=ResNet50_224x224.prototxt --output=prob  --batch=16 --useDLACore=0  --allowGPUFallback  --best --workspace=4096

echo "ssd-mobilenet-v1 GPU"
/usr/src/tensorrt/bin/trtexec ${common}  --onnx=ssd-mobilenet-v1-bs1.onnx   --best --workspace=4096    
/usr/src/tensorrt/bin/trtexec ${common}  --onnx=ssd-mobilenet-v1-bs2.onnx   --best --workspace=4096  
/usr/src/tensorrt/bin/trtexec ${common}  --onnx=ssd-mobilenet-v1-bs4.onnx   --best --workspace=4096  
/usr/src/tensorrt/bin/trtexec ${common}  --onnx=ssd-mobilenet-v1-bs8.onnx   --best --workspace=4096  
/usr/src/tensorrt/bin/trtexec ${common}  --onnx=ssd-mobilenet-v1-bs16.onnx   --best --workspace=4096 
