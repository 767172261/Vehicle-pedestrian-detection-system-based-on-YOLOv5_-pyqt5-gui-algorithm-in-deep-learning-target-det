深度学习目标检测中基于 YOLOv5/+pyqt5 gui算法的车辆行人检测系统识别车辆，自行车，行人红绿灯等
@[toc]
，可识别车辆，自行车，行人，红绿灯多种类型目标
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c5363a4ed17549bd8b3a8cbd5351df54.png)

构建基于YOLOv5的车辆和行人检测系统，并可用GUI界面调用，可以分为以下几个步骤：

1. **数据准备与模型训练**
2. **模型推理**
3. **GUI界面设计**
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b9e3b6283c604fe89023d7048c6f881c.png)

### 1. 数据准备与模型训练

#### 数据集准备
确保包含车辆、自行车、行人、红绿灯等标注的数据集。数据集应该分为训练集、验证集和测试集。

#### 标注工具
使用如LabelImg或VOC格式进行标注。

#### 数据集转换
将标注好的数据集转换为YOLO格式。

```python
import xml.etree.ElementTree as ET
import os

def convert_voc_to_yolo(voc_dir, yolo_dir, classes):
    if not os.path.exists(yolo_dir):
        os.makedirs(yolo_dir)
    
    for xml_file in os.listdir(voc_dir):
        if not xml_file.endswith('.xml'):
            continue
        
        tree = ET.parse(os.path.join(voc_dir, xml_file))
        root = tree.getroot()
        
        image_name = root.find('filename').text
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)
        
        with open(os.path.join(yolo_dir, image_name.replace('.jpg', '.txt')), 'w') as f:
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                class_id = classes.index(class_name)
                
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                x_center = (xmin + xmax) / 2.0 / width
                y_center = (ymin + ymax) / 2.0 / height
                w = (xmax - xmin) / float(width)
                h = (ymax - ymin) / float(height)
                
                f.write(f"{class_id} {x_center} {y_center} {w} {h}\n")

# 示例调用
convert_voc_to_yolo('path/to/voc_labels', 'path/to/yolo_labels', ['car', 'bike', 'person', 'traffic_light'])
```

#### 训练模型
使用YOLOv5进行模型训练。

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt

# 修改data.yaml文件以匹配你的数据集
nano data/my_data.yaml

# 训练模型
python train.py --img 640 --batch 16 --epochs 100 --data my_data.yaml --weights yolov5s.pt
```

### 2. 模型推理

编写推理代码，用于加载模型并进行预测。

```python
from ultralytics import YOLO

model = YOLO('best.pt')  # 加载训练好的模型

def detect_objects(image_path):
    results = model(image_path)
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            r = box.xyxy[0].astype(int)
            conf = box.conf
            cls = int(box.cls)
            label = f'{model.names[cls]} {conf:.2f}'
            cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), (255, 0, 0), 2)
            cv2.putText(image, label, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image
```

### 3. GUI界面设计

使用PyQt5设计GUI界面。

#### 设计UI
使用Qt Designer设计界面，并保存为`.ui`文件。

#### 编写Python代码

```python
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi
from ultralytics import YOLO

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi('yolov5.ui', self)

        self.model = YOLO('best.pt')  # 加载训练好的模型

        self.open_button.clicked.connect(self.open_image)
        self.detect_button.clicked.connect(self.detect_objects)
        self.save_button.clicked.connect(self.save_result)

    def open_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)", options=options)
        if file_name:
            self.image_path = file_name
            self.update_image()

    def update_image(self):
        image = cv2.imread(self.image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qimage = QImage(image_rgb.data, image_rgb.shape[1], image_rgb.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap)

    def detect_objects(self):
        image = cv2.imread(self.image_path)
        results = self.model(image)
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                r = box.xyxy[0].astype(int)
                conf = box.conf
                cls = int(box.cls)
                label = f'{self.model.names[cls]} {conf:.2f}'
                cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), (255, 0, 0), 2)
                cv2.putText(image, label, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        self.update_image()

    def save_result(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Image Files (*.png *.jpg *.jpeg)", options=options)
        if file_name:
            cv2.imwrite(file_name, self.image)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7283d390e69a463993900b570adcc7a5.png)

### 总结

构建一个基于YOLOv5的车辆和行人检测系统，并实现一个用户友好的GUI界面。

### 提高精度的方法

- **数据增强**：增加更多的数据样本，并使用数据增强技术。
- **超参数调整**：调整学习率、批量大小等超参数。
- **模型选择**：尝试不同的YOLOv5版本（如YOLOv5m、YOLOv5l）。
- **后处理**：优化NMS阈值等后处理参数。


