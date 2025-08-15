import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, Any, Tuple, Optional, List
import cv2
import numpy as np
import random
import pickle
import hashlib
from pathlib import Path

from .base_dataset import BaseDataset
from .registry import DatasetRegistry
from ..utils.logger import get_logger
from .preprocessing import PlatePreprocessor

@DatasetRegistry.register('augmented_plate')
class AugmentedPlateDataset(BaseDataset):
    """增强版盘子分类数据集，支持多重裁剪和分割增强，带缓存机制"""
    
    def __init__(self, config: Dict[str, Any], is_training: bool = True):
        """初始化
        
        Args:
            config: 数据集配置
            is_training: 是否为训练集
        """
        super().__init__(config, is_training)
        self.logger = get_logger(__name__)
        
        # 缓存配置
        self.use_cache = config.get('use_cache', True)
        self.cache_in_memory = config.get('cache_in_memory', True)
        self.cache_dir = config.get('cache_dir', os.path.join(self.data_dir, '.cache'))
        
        # 初始化预处理器
        self.use_preprocessing = config.get('use_preprocessing', False)
        if self.use_preprocessing:
            self.preprocessor = PlatePreprocessor()
            
        # 多重裁剪增强配置
        self.use_crop_augmentation = config.get('use_crop_augmentation', False) and is_training
        self.crop_steps = config.get('crop_steps', 5)  # 裁剪步长
        self.max_crop_ratio = config.get('max_crop_ratio', 1/3)  # 最大裁剪比例
        self.quadrant_split = config.get('quadrant_split', False)  # 是否进行四象限分割
        
        # 缓存相关变量
        self.cached_images = {}  # 内存缓存
        self.cache_loaded = False
        
        self.transform = self._get_transforms()
        self._load_data()
        
        # 如果启用多重裁剪增强，扩展数据集
        if self.use_crop_augmentation:
            self._expand_dataset_with_crops()
            
        # 初始化缓存
        if self.use_cache:
            self._initialize_cache()
    
    def _get_cache_key(self) -> str:
        """生成缓存键，基于配置和数据路径"""
        config_str = str(sorted(self.config.items()))
        data_info = f"{self.data_dir}_{self.is_training}_{len(self.image_paths)}"
        cache_content = f"{config_str}_{data_info}"
        return hashlib.md5(cache_content.encode()).hexdigest()
    
    def _get_cache_path(self) -> Path:
        """获取缓存文件路径"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        
        cache_key = self._get_cache_key()
        split_name = "train" if self.is_training else "test"
        cache_filename = f"augmented_plate_{split_name}_{cache_key}.pkl"
        return Path(self.cache_dir) / cache_filename
    
    def _initialize_cache(self):
        """初始化缓存系统"""
        cache_path = self._get_cache_path()
        
        # 尝试加载现有缓存
        if cache_path.exists():
            self.logger.info(f"尝试加载缓存文件: {cache_path}")
            try:
                self._load_cache_from_disk(cache_path)
                return
            except Exception as e:
                self.logger.warning(f"加载缓存失败: {e}，将重新生成缓存")
        
        # 生成新缓存
        self.logger.info("开始生成图像缓存...")
        self._generate_cache()
        
        # 保存缓存到磁盘
        if self.use_cache:
            self._save_cache_to_disk(cache_path)
    
    def _load_cache_from_disk(self, cache_path: Path):
        """从磁盘加载缓存"""
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # 验证缓存数据完整性
        if len(cache_data) != len(self.image_paths):
            raise ValueError("缓存数据长度与当前数据集不匹配")
        
        if self.cache_in_memory:
            self.cached_images = cache_data
            self.logger.info(f"成功加载 {len(cache_data)} 张图像到内存缓存")
        else:
            # 不加载到内存，但标记缓存可用
            self.cache_loaded = True
            self.logger.info(f"缓存文件可用，包含 {len(cache_data)} 张图像")
    
    def _save_cache_to_disk(self, cache_path: Path):
        """保存缓存到磁盘"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.cached_images, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.logger.info(f"缓存已保存到: {cache_path}")
        except Exception as e:
            self.logger.error(f"保存缓存失败: {e}")
    
    def _generate_cache(self):
        """生成图像缓存"""
        total_images = len(self.image_paths)
        
        for idx in range(total_images):
            if idx % 100 == 0:
                self.logger.info(f"处理缓存进度: {idx}/{total_images}")
            
            try:
                processed_image = self._load_and_process_image(idx)
                
                if self.cache_in_memory:
                    self.cached_images[idx] = processed_image
                    
            except Exception as e:
                self.logger.error(f"处理图像 {self.image_paths[idx]} 时出错: {e}")
                # 使用默认图像作为占位符
                if self.cache_in_memory:
                    self.cached_images[idx] = self._get_default_image()
        
        self.cache_loaded = True
        self.logger.info(f"缓存生成完成，共处理 {total_images} 张图像")
    
    def _load_and_process_image(self, idx: int) -> Image.Image:
        """加载并处理单张图像"""
        img_path = self.image_paths[idx]
        crop_info = self.crop_info[idx]
        
        if self.use_preprocessing:
            # 使用OpenCV读取图像进行预处理
            img_cv = cv2.imread(img_path)
            if img_cv is None:
                raise ValueError(f"无法读取图像: {img_path}")
            
            # 应用预处理
            processed_img = self.preprocessor.process_image(img_cv, self.is_training)
            
            # 转换为PIL图像
            image = Image.fromarray(processed_img)
        else:
            # 常规PIL读取
            image = Image.open(img_path).convert('RGB')
        
        # 应用裁剪增强
        if self.use_crop_augmentation:
            image = self._apply_crop_augmentation(image, crop_info)
        
        return image
    
    def _get_default_image(self) -> Image.Image:
        """获取默认占位图像"""
        # 创建一个默认的RGB图像
        return Image.new('RGB', (224, 224), color=(128, 128, 128))
    
    def _get_transforms(self) -> transforms.Compose:
        """获取数据变换
        
        Returns:
            数据变换组合
        """
        transform_config = self.config.get('transforms', {})
        
        transform_list = []
        
        # 如果是训练集，添加高级数据增强
        if self.is_training:
            # 透视变换
            if transform_config.get('random_perspective', {}).get('enable', False):
                perspective_config = transform_config['random_perspective']
                transform_list.append(transforms.RandomPerspective(
                    distortion_scale=perspective_config.get('distortion_scale', 0.09),
                    p=perspective_config.get('p', 0.75),
                    interpolation=3,
                    fill=0
                ))
        
        # 调整大小
        if 'resize' in transform_config:
            size = transform_config['resize']
            transform_list.append(transforms.Resize(size))
        
        # 中心裁剪
        if 'center_crop' in transform_config:
            size = transform_config['center_crop']
            transform_list.append(transforms.CenterCrop(size))
        
        # 如果是训练集，添加更多数据增强
        if self.is_training:
            # 颜色抖动（更强的参数）
            color_jitter_config = transform_config.get('color_jitter', {})
            if color_jitter_config.get('enable', True):
                hue_range = color_jitter_config.get('hue', (-0.5, 0.5))
                transform_list.append(transforms.ColorJitter(
                    brightness=color_jitter_config.get('brightness', 0.2),
                    contrast=color_jitter_config.get('contrast', 0.2),
                    saturation=color_jitter_config.get('saturation', 0.1),
                    hue=hue_range
                ))
            
            # 翻转
            if transform_config.get('horizontal_flip', True):
                transform_list.append(transforms.RandomHorizontalFlip())
            if transform_config.get('vertical_flip', False):
                transform_list.append(transforms.RandomVerticalFlip())
                
            # 旋转
            rotation_degrees = transform_config.get('rotation', 15)
            if rotation_degrees > 0:
                transform_list.append(transforms.RandomRotation(rotation_degrees))
        
        # 转换为张量
        transform_list.append(transforms.ToTensor())
        
        # 标准化
        if 'normalize' in transform_config:
            mean = transform_config['normalize'].get('mean', [0.485, 0.456, 0.406])
            std = transform_config['normalize'].get('std', [0.229, 0.224, 0.225])
            transform_list.append(transforms.Normalize(mean, std))
        
        return transforms.Compose(transform_list)
    
    def _load_data(self) -> None:
        """加载数据"""
        # 获取盘子数据的根目录
        plate_dir = os.path.join(self.data_dir, self.config.get('data_name', 'plate'))
        
        if not os.path.exists(plate_dir):
            raise FileNotFoundError(f"盘子数据集目录不存在: {plate_dir}")
        
        # 创建图像路径和标签列表
        self.image_paths = []
        self.labels = []
        self.crop_info = []  # 存储裁剪信息
        
        # 检查是否使用分割文件夹结构
        use_split_folders = self.config.get('use_split_folders', False)
        
        if use_split_folders:
            self._load_data_from_split_folders(plate_dir)
        else:
            self._load_data_from_class_folders(plate_dir)
        
        # 检查是否有足够的数据
        if len(self.image_paths) == 0:
            mode = "训练" if self.is_training else "测试"
            raise ValueError(f"没有找到{mode}数据。请确保数据目录结构正确。")
        
        # 初始化裁剪信息（原始图像不裁剪）
        self.crop_info = [{'type': 'original', 'params': None} for _ in range(len(self.image_paths))]
        
        # 打印数据集信息
        if self.is_training or -1 not in self.labels:
            self.logger.info(f"{'训练' if self.is_training else '测试'}集加载完成: "
                  f"总共 {len(self.image_paths)} 张图片, "
                  f"干净盘子: {self.labels.count(0)}, "
                  f"脏盘子: {self.labels.count(1)}")
        else:
            self.logger.info(f"测试集加载完成: 总共 {len(self.image_paths)} 张图片（无标签）")
    
    def _load_data_from_split_folders(self, plate_dir: str) -> None:
        """从分割文件夹结构加载数据 (train/val/test)"""
        # 确定要加载的文件夹
        if self.is_training:
            split_folder = 'train'
        else:
            if os.path.exists(os.path.join(plate_dir, 'test')):
                split_folder = 'test'
            elif os.path.exists(os.path.join(plate_dir, 'val')):
                split_folder = 'val'
            else:
                raise FileNotFoundError(f"在 {plate_dir} 中找不到 'test' 或 'val' 文件夹")
        
        split_dir = os.path.join(plate_dir, split_folder)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"分割文件夹不存在: {split_dir}")
        
        # 如果是测试集，直接读取test文件夹下的所有图像文件
        if not self.is_training and split_folder == 'test':
            all_images = []
            for fname in os.listdir(split_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_images.append(os.path.join(split_dir, fname))
            
            all_images.sort()
            self.image_paths.extend(all_images)
            self.labels.extend([-1] * len(all_images))
            
            self.logger.info(f"从 {split_folder} 文件夹直接加载所有图像文件")
            return
        
        # 训练集或验证集：按照分类子文件夹加载
        # 加载干净盘子图像
        clean_dir = os.path.join(split_dir, 'clean_dish')
        if os.path.exists(clean_dir):
            clean_images = [os.path.join(clean_dir, fname) for fname in os.listdir(clean_dir) 
                            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
            clean_images.sort()
            self.image_paths.extend(clean_images)
            self.labels.extend([0] * len(clean_images))
        
        # 加载脏盘子图像
        dirty_dir = os.path.join(split_dir, 'dirty_dish')
        if os.path.exists(dirty_dir):
            dirty_images = [os.path.join(dirty_dir, fname) for fname in os.listdir(dirty_dir) 
                           if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
            dirty_images.sort()
            self.image_paths.extend(dirty_images)
            self.labels.extend([1] * len(dirty_images))
        
        self.logger.info(f"从 {split_folder} 文件夹加载数据")
    
    def _load_data_from_class_folders(self, plate_dir: str) -> None:
        """从类别文件夹结构加载数据"""
        train_ratio = self.config.get('train_ratio', 0.8)
        
        # 加载干净盘子图像
        clean_dir = os.path.join(plate_dir, 'clean_dish')
        if os.path.exists(clean_dir):
            clean_images = [os.path.join(clean_dir, fname) for fname in os.listdir(clean_dir) 
                            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
            clean_images.sort()
            
            split_idx = int(len(clean_images) * train_ratio)
            if self.is_training:
                selected_clean = clean_images[:split_idx]
            else:
                selected_clean = clean_images[split_idx:]
            
            self.image_paths.extend(selected_clean)
            self.labels.extend([0] * len(selected_clean))
        
        # 加载脏盘子图像
        dirty_dir = os.path.join(plate_dir, 'dirty_dish')
        if os.path.exists(dirty_dir):
            dirty_images = [os.path.join(dirty_dir, fname) for fname in os.listdir(dirty_dir) 
                           if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
            dirty_images.sort()
            
            split_idx = int(len(dirty_images) * train_ratio)
            if self.is_training:
                selected_dirty = dirty_images[:split_idx]
            else:
                selected_dirty = dirty_images[split_idx:]
            
            self.image_paths.extend(selected_dirty)
            self.labels.extend([1] * len(selected_dirty))
    
    def _expand_dataset_with_crops(self) -> None:
        """使用多重裁剪扩展数据集"""
        original_paths = self.image_paths.copy()
        original_labels = self.labels.copy()
        original_crop_info = self.crop_info.copy()
        
        # 为每张原始图像添加多种裁剪版本
        for i, (img_path, label) in enumerate(zip(original_paths, original_labels)):
            # 跳过测试集的无标签数据
            if label == -1:
                continue
                
            # 添加多重裁剪
            self._add_multiple_crops(img_path, label)
            
            # 添加四象限分割（如果启用）
            if self.quadrant_split:
                self._add_quadrant_splits(img_path, label)
        
        self.logger.info(f"多重裁剪增强后数据集大小: {len(self.image_paths)} 张图片")
    
    def _add_multiple_crops(self, img_path: str, label: int) -> None:
        """为单张图像添加多种尺寸的裁剪"""
        # 读取图像获取尺寸信息
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                min_size = min(w, h)
        except Exception as e:
            self.logger.warning(f"无法读取图像 {img_path}: {e}")
            return
        
        # 计算裁剪尺寸范围
        max_crop_size = int(min_size * (1 - self.max_crop_ratio))
        min_crop_size = max(50, max_crop_size // 3)  # 最小裁剪尺寸
        
        # 生成多个裁剪尺寸
        for i in range(self.crop_steps, int(min_size * self.max_crop_ratio), self.crop_steps):
            crop_size = min_size - i
            if crop_size >= min_crop_size:
                self.image_paths.append(img_path)
                self.labels.append(label)
                self.crop_info.append({
                    'type': 'crop',
                    'params': {'crop_size': crop_size}
                })
    
    def _add_quadrant_splits(self, img_path: str, label: int) -> None:
        """为单张图像添加四象限分割"""
        quadrants = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        
        for quadrant in quadrants:
            self.image_paths.append(img_path)
            self.labels.append(label)
            self.crop_info.append({
                'type': 'quadrant',
                'params': {'quadrant': quadrant}
            })
    
    def _apply_crop_augmentation(self, image: Image.Image, crop_info: Dict) -> Image.Image:
        """应用裁剪增强"""
        if crop_info['type'] == 'original':
            return image
        elif crop_info['type'] == 'crop':
            # 中心裁剪
            crop_size = crop_info['params']['crop_size']
            w, h = image.size
            left = (w - crop_size) // 2
            top = (h - crop_size) // 2
            right = left + crop_size
            bottom = top + crop_size
            return image.crop((left, top, right, bottom))
        elif crop_info['type'] == 'quadrant':
            # 四象限分割
            quadrant = crop_info['params']['quadrant']
            w, h = image.size
            half_w, half_h = w // 2, h // 2
            
            if quadrant == 'top_left':
                return image.crop((0, 0, half_w, half_h))
            elif quadrant == 'top_right':
                return image.crop((half_w, 0, w, half_h))
            elif quadrant == 'bottom_left':
                return image.crop((0, half_h, half_w, h))
            elif quadrant == 'bottom_right':
                return image.crop((half_w, half_h, w, h))
        
        return image
    
    def clear_cache(self):
        """清理内存缓存"""
        if hasattr(self, 'cached_images'):
            self.cached_images.clear()
            self.logger.info("内存缓存已清理")
    
    def __len__(self) -> int:
        """获取数据集长度"""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取数据项"""
        try:
            # 优先使用缓存的图像
            if self.use_cache and self.cache_in_memory and idx in self.cached_images:
                image = self.cached_images[idx]
            else:
                # 实时加载和处理图像
                image = self._load_and_process_image(idx)
                
        except Exception as e:
            self.logger.error(f"加载图像索引 {idx} 时出错: {e}")
            # 如果出错，返回数据集中的另一个图像
            return self.__getitem__((idx + 1) % len(self))
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 获取标签
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label 