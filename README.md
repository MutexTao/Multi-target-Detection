# Multi-target-Detection

环境配置：安装 ultralytics 软件包，pip install ultralytics或通过运行 pip install -U ultralytics.请访问Python Package Index (PyPI)，了解更多有关 ultralytics 包装 https://pypi.org/project/ultralytics/.python版本推荐python3.8

更多操作请参考https://docs.ultralytics.com/zh/quickstart/

该项目为课程作业项目，使用VOC数据集，需要在../dataset/中进行下载，具体下载地址请参考yolo官方，将yolov10的backbone替换为RT-DETR的PP-HGNetv2，取得了涨点，并且推理速度也高于RT-DETR，若要使用我们的yolov10魔改的HG版本，请在你的conda环境中找到对应的env，比如我的conda虚拟环境叫yolo，那么进入路径yolo/lib/python3.8/site-packages/ultralytics/cfg/models/v10，创建一个yolov10HG，内容为：

```
# Ultralytics YOLO 🚀, AGPL-3.0 license
# YOLOv10 object detection model. For Usage examples see https://docs.ultralytics.com/tasks/detect
 
# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov10n.yaml' will call yolov10.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024]
 
backbone:
  # [from, repeats, module, args]
  - [-1, 1, HGStem, [32, 48]]  # 0-P2/4
  - [-1, 6, HGBlock, [48, 128, 3]]  # stage 1
 
  - [-1, 1, DWConv, [128, 3, 2, 1, False]]  # 2-P3/8
  - [-1, 6, HGBlock, [96, 512, 3]]   # stage 2
 
  - [-1, 1, DWConv, [512, 3, 2, 1, False]]  # 4-P3/16
  - [-1, 6, HGBlock, [192, 1024, 5, True, False]]  # cm, c2, k, light, shortcut
  - [-1, 6, HGBlock, [192, 1024, 5, True, True]]
  - [-1, 6, HGBlock, [192, 1024, 5, True, True]]  # stage 3
 
  - [-1, 1, DWConv, [1024, 3, 2, 1, False]]  # 8-P4/32
  - [-1, 6, HGBlock, [384, 2048, 5, True, False]]  # stage 4
 
  - [-1, 1, SPPF, [1024, 5]] # 10
  - [-1, 1, PSA, [1024]] # 11
 
# YOLOv10.0n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 12
  - [[-1, 7], 1, Concat, [1]] # cat backbone P4
  - [-1, 3, C2f, [512]] # 14
 
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 3], 1, Concat, [1]] # cat backbone P3
  - [-1, 3, C2f, [256]] # 17 (P3/8-small)
 
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 14], 1, Concat, [1]] # cat head P4
  - [-1, 3, C2f, [512]] # 20 (P4/16-medium)
 
  - [-1, 1, SCDown, [512, 3, 2]]
  - [[-1, 11], 1, Concat, [1]] # cat head P5
  - [-1, 3, C2fCIB, [1024, True, True]] # 23 (P5/32-large)
 
  - [[17, 20, 23], 1, v10Detect, [nc]] # Detect(P3, P4, P5)
```

最后运行yolov10-HG.py即可，若需要修改和教程请参考https://docs.ultralytics.com/zh/modes/train
