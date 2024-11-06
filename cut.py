import cv2
import numpy as np
from ultralytics import YOLO

class CutAdhesionDetector:
    def __init__(self, model_path: str, regions: list, width=400, height=300):
        """初始化 YOLO 模型和裁剪参数"""
        self.model = YOLO(model_path)
        self.regions = regions  # 5个高风险粘连区域中心点坐标

        # 裁剪区域长宽
        self.width = width
        self.height = height

    def crop_image_regions(self, image: np.ndarray) -> list:
        """裁剪指定区域"""
        cropped_images = []

        for x, y in self.regions:
            x1, y1 = int(x - self.width / 2), int(y - self.height / 2)  # 左上角
            x2, y2 = int(x + self.width / 2), int(y + self.height / 2)  # 右下角

            cropped_image = image[y1:y2, x1:x2]
            cropped_images.append(cropped_image)

        return cropped_images

    def infer_yolo(self, images: list) -> list:
        """对图像进行推断"""
        results = []
        for img in images:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
            result = self.model(img_rgb)
            results.append(result)
        return results

    def detect(self, im: np.ndarray) -> bool:
        """cut粘连分类处理"""

        cropped_images = self.crop_image_regions(im)

        detections = self.infer_yolo(cropped_images)

        results=[]
        for outputs,(x,y) in zip(detections,self.regions):
            label = outputs[0].probs.top1
            confidence = round(float(outputs[0].probs.top1conf),2)
            xywh=[x,y,self.width,self.height]
            if label==1 and confidence>0.6:
                results.append([label,confidence,xywh])
        return results


# 示例用法
if __name__ == "__main__":
    image_path = r'C:\Users\Administrator\workspace\project\yolov8l_cut\datasets\images\train\240904_00059.bmp'
    im = cv2.imread(image_path)
    regions = [
        (800, 700),
        (1500, 640),
        (820, 1480),
        (1600, 950),
        (1200, 1150),
        (1630, 1370),
        (1470, 1480)
    ]

    # 定义裁剪宽度和高度
    w, h = 400, 300
    modelpath=r"models\cut\cutadhesion_best.pt"
    # 创建处理器实例并处理图像
    processor = CutAdhesionDetector(model_path=modelpath,regions=regions, width=w,
                                   height=h)
    result = processor.detect(im)
    print("检测结果:", result)

