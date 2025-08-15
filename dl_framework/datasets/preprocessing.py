import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PlatePreprocessor:
    """盘子图像预处理器，包含背景移除和圆形检测功能"""
    
    def __init__(self):
        self.x00 = 0
        self.y00 = 0
        self.r00 = 0
        
    def remove_background(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """使用GrabCut算法移除背景
        
        Args:
            img: 输入图像 (BGR格式)
            
        Returns:
            处理后的图像和掩码
        """
        mainRectSize = 0.08
        fgSize = 0.01
        
        # 转换为RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        new_h, new_w = img_rgb.shape[:2]
        
        # 创建掩码
        mask = np.zeros(img_rgb.shape[:2], np.uint8)
        
        # 计算背景和前景矩形
        bg_w = round(new_w * mainRectSize)
        bg_h = round(new_h * mainRectSize)
        bg_rect = (bg_w, bg_h, new_w - bg_w, new_h - bg_h)
        
        fg_w = round(new_w * (1 - fgSize) / 2)
        fg_h = round(new_h * (1 - fgSize) / 2)
        fg_rect = (fg_w, fg_h, new_w - fg_w, new_h - fg_h)
        
        # 设置前景区域
        cv2.rectangle(mask, fg_rect[:2], fg_rect[2:4], color=cv2.GC_FGD, thickness=-1)
        
        # GrabCut算法
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        cv2.grabCut(img_rgb, mask, bg_rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
        cv2.rectangle(mask, bg_rect[:2], bg_rect[2:4], color=cv2.GC_PR_BGD, thickness=bg_w * 3)
        cv2.grabCut(img_rgb, mask, bg_rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_MASK)
        
        # 创建最终掩码
        mask_result = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
        masked = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_result)
        masked[mask_result < 2] = [255, 255, 255]
        
        return masked, mask_result
    
    def find_coordinates(self, mask: np.ndarray) -> Tuple[int, int]:
        """通过图像矩找到中心坐标
        
        Args:
            mask: 二值掩码
            
        Returns:
            中心坐标 (x, y)
        """
        ret, thresh = cv2.threshold(mask, 235, 255, 0)
        M = cv2.moments(thresh)
        
        if M["m00"] != 0:
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
        else:
            # 如果无法计算矩，使用图像中心
            h, w = mask.shape[:2]
            x, y = w // 2, h // 2
            
        return x, y
    
    def find_circle(self, img: np.ndarray, center_x: int, center_y: int) -> Tuple[int, int, int]:
        """使用HoughCircles检测圆形
        
        Args:
            img: 输入图像
            center_x: 预估中心x坐标
            center_y: 预估中心y坐标
            
        Returns:
            圆心坐标和半径 (x, y, r)
        """
        # 增强对比度
        enhanced = cv2.convertScaleAbs(img, alpha=1.2, beta=0.0)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
        
        # 检测圆形
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 10,
            param1=10, param2=5, minRadius=40, maxRadius=250
        )
        
        best_x, best_y, best_r = center_x, center_y, 0
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # 找到最接近预估中心的圆
            for x, y, r in circles:
                if ((center_x - 15) < x < (center_x + 15)) and ((center_y - 15) < y < (center_y + 15)):
                    if r > best_r:
                        best_x, best_y, best_r = x, y, r
        
        # 如果没找到合适的圆，使用轮廓检测作为备选
        if best_r == 0:
            try:
                gray_thresh = cv2.threshold(gray, 235, 255, 0)[1]
                contours, _ = cv2.findContours(gray_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    (x, y), r = cv2.minEnclosingCircle(c)
                    best_x, best_y, best_r = int(x), int(y), int(r)
            except Exception as e:
                logger.warning(f"轮廓检测失败: {e}")
                # 使用默认半径
                best_r = min(img.shape[:2]) // 4
        
        return best_x, best_y, best_r
    
    def crop_plate(self, img: np.ndarray, center_x: int, center_y: int, radius: int, 
                   crop_factor: float = 1.42) -> np.ndarray:
        """根据检测到的圆形裁剪盘子区域
        
        Args:
            img: 输入图像
            center_x: 圆心x坐标
            center_y: 圆心y坐标  
            radius: 圆半径
            crop_factor: 裁剪系数，控制裁剪区域大小
            
        Returns:
            裁剪后的图像
        """
        crop_radius = int(crop_factor * radius / 2)
        
        # 计算裁剪边界
        y1 = max(0, center_y - crop_radius)
        y2 = min(img.shape[0], center_y + crop_radius)
        x1 = max(0, center_x - crop_radius)
        x2 = min(img.shape[1], center_x + crop_radius)
        
        # 裁剪图像
        cropped = img[y1:y2, x1:x2]
        
        return cropped
    
    def process_image(self, img: np.ndarray, for_training: bool = True) -> np.ndarray:
        """完整的图像预处理流程
        
        Args:
            img: 输入图像 (BGR格式)
            for_training: 是否用于训练（影响输出格式）
            
        Returns:
            处理后的图像
        """
        try:
            # 1. 移除背景
            masked_img, mask = self.remove_background(img)
            
            # 2. 找到中心坐标
            center_x, center_y = self.find_coordinates(mask)
            
            # 3. 检测圆形
            circle_x, circle_y, radius = self.find_circle(masked_img, center_x, center_y)
            
            # 4. 裁剪图像
            cropped = self.crop_plate(masked_img, circle_x, circle_y, radius)
            
            return cropped
            
        except Exception as e:
            logger.error(f"图像预处理失败: {e}")
            # 返回原图像的中心裁剪作为备选
            h, w = img.shape[:2]
            size = min(h, w)
            center_y, center_x = h // 2, w // 2
            y1 = center_y - size // 2
            y2 = center_y + size // 2
            x1 = center_x - size // 2  
            x2 = center_x + size // 2
            return cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB) 