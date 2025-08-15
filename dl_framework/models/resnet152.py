import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Union, Optional
import torchvision.models as tv_models

from .base_model import BaseModel
from .registry import ModelRegistry
from ..utils.logger import get_logger


@ModelRegistry.register('resnet152')
class ResNet152(BaseModel):
    """基于 torchvision 的 ResNet152 模型包装器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化 ResNet152 模型
        
        Args:
            config: 模型配置，包含以下参数：
                - num_classes: 分类数量，默认为 1000
                - pretrained: 是否使用预训练权重，默认为 False
                - weights: 指定预训练权重类型
                - freeze_backbone: 是否冻结主干网络，默认为 False
                - dropout: dropout 概率，默认为 0.0
        """
        super(ResNet152, self).__init__(config)
        self.logger = get_logger(__name__)
        
        # 从配置中获取参数
        self.num_classes = config.get('num_classes', 1000)
        self.pretrained = config.get('pretrained', False)
        self.weights = config.get('weights', None)
        self.freeze_backbone = config.get('freeze_backbone', False)
        self.dropout_prob = config.get('dropout', 0.0)
        
        # 创建 torchvision 的 ResNet152 模型
        self._create_model()
    
    def _create_model(self):
        """创建 torchvision ResNet152 模型"""
        # 确定权重类型
        weights = None
        if self.pretrained or self.weights:
            if self.weights:
                if self.weights.upper() in ['DEFAULT', 'IMAGENET1K_V2']:
                    weights = tv_models.ResNet152_Weights.IMAGENET1K_V2
                elif self.weights.upper() == 'IMAGENET1K_V1':
                    weights = tv_models.ResNet152_Weights.IMAGENET1K_V1
                else:
                    self.logger.warning(f"未知的权重类型: {self.weights}，使用默认权重")
                    weights = tv_models.ResNet152_Weights.IMAGENET1K_V2
            else:
                weights = tv_models.ResNet152_Weights.IMAGENET1K_V2
        
        # 创建模型
        self.model = tv_models.resnet152(weights=weights)
        
        # 冻结主干网络参数
        if self.freeze_backbone:
            self.logger.info("冻结ResNet152主干网络参数")
            for param in self.model.parameters():
                param.requires_grad = False
        
        # 替换最后的全连接层
        in_features = self.model.fc.in_features
        
        if self.dropout_prob > 0:
            # 添加dropout层
            self.model.fc = nn.Sequential(
                nn.Dropout(self.dropout_prob),
                nn.Linear(in_features, self.num_classes)
            )
        else:
            # 直接替换全连接层
            self.model.fc = nn.Linear(in_features, self.num_classes)
        
        # 如果冻结了主干网络，确保新的分类器层可以训练
        if self.freeze_backbone:
            for param in self.model.fc.parameters():
                param.requires_grad = True
        
        self.logger.info(f"ResNet152 模型创建完成，分类数量: {self.num_classes}")
        self.logger.info(f"预训练权重: {'是' if weights else '否'}")
        self.logger.info(f"冻结主干网络: {'是' if self.freeze_backbone else '否'}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, channels, height, width)
            
        Returns:
            输出张量，形状为 (batch_size, num_classes)
        """
        return self.model(x)
    
    def get_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算损失
        
        Args:
            outputs: 模型输出，形状为[batch_size, num_classes]
            targets: 目标标签，形状为[batch_size]
            
        Returns:
            损失值
        """
        # 对于二分类任务，使用交叉熵损失
        return F.cross_entropy(outputs, targets)
    
    def get_feature_extractor(self) -> nn.Module:
        """获取特征提取器（不包含分类器）
        
        Returns:
            特征提取器模块
        """
        # 创建一个新的模型，去掉最后的全连接层
        feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        return feature_extractor
    
    def get_classifier(self) -> nn.Module:
        """获取分类器部分
        
        Returns:
            分类器模块
        """
        return self.model.fc
    
    def freeze_backbone(self):
        """冻结主干网络参数"""
        self.logger.info("冻结ResNet152主干网络参数")
        for name, param in self.model.named_parameters():
            if 'fc' not in name:  # 不冻结分类器层
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """解冻主干网络参数"""
        self.logger.info("解冻ResNet152主干网络参数")
        for param in self.model.parameters():
            param.requires_grad = True
    
    def get_trainable_parameters(self):
        """获取可训练参数"""
        return [p for p in self.model.parameters() if p.requires_grad]
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息
        
        Returns:
            包含模型信息的字典
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': 'ResNet152',
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'freeze_backbone': self.freeze_backbone,
            'pretrained': self.pretrained,
            'dropout': self.dropout_prob
        } 