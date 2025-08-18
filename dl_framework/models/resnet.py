import torch
import torch.nn as nn
from typing import Dict, Any
import torchvision.models as tv_models

from .base_model import BaseModel
from .registry import ModelRegistry
from ..utils.logger import get_logger


@ModelRegistry.register('resnet50')
class ResNet50(BaseModel):
    """基于 torchvision 的 ResNet-50 模型包装器"""

    def __init__(self, config: Dict[str, Any]):
        """初始化 ResNet-50 模型
        
        Args:
            config: 模型配置，包含以下参数：
                - num_classes: 分类数量，默认为 1000
                - pretrained: 是否使用预训练权重，默认为 False
                - weights: 指定预训练权重类型，可选 'IMAGENET1K_V1', 'IMAGENET1K_V2' 或 'DEFAULT'
                - freeze_backbone: 是否冻结主干网络，默认为 False
        """
        super(ResNet50, self).__init__(config)
        self.logger = get_logger(__name__)
        
        # 从配置中获取参数
        self.num_classes = config.get('num_classes', 1000)
        self.pretrained = config.get('pretrained', False)
        self.weights = config.get('weights', None)
        self.freeze = config.get('freeze_backbone', False)
        
        # 创建 torchvision 的 ResNet-50 模型
        self._create_model()

    def _create_model(self):
        """创建 torchvision ResNet-50 模型"""
        # 确定权重类型
        weights = None
        if self.pretrained or self.weights:
            if self.weights:
                if self.weights.upper() == 'IMAGENET1K_V1':
                    weights = tv_models.ResNet50_Weights.IMAGENET1K_V1
                elif self.weights.upper() == 'IMAGENET1K_V2':
                    weights = tv_models.ResNet50_Weights.IMAGENET1K_V2
                elif self.weights.upper() == 'DEFAULT':
                    weights = tv_models.ResNet50_Weights.DEFAULT
                else:
                    # 尝试直接使用字符串
                    weights = self.weights
            else:
                weights = tv_models.ResNet50_Weights.DEFAULT
        
        # 创建模型
        self.model = tv_models.resnet50(
            weights=weights
        )
        
        # 修改分类层
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, self.num_classes)
        )
        
        if self.freeze:
            self.freeze_backbone()
            
        # 打印加载信息
        if weights:
            self.logger.info(f"成功加载 ResNet-50 预训练权重: {weights}")
            if self.num_classes != 1000:
                self.logger.info(f"分类层已调整为 {self.num_classes} 类")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.model(x)

    def get_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算损失"""
        if hasattr(self, 'loss') and self.loss is not None:
            return super().get_loss(outputs, targets)
        
        criterion = nn.CrossEntropyLoss()
        return criterion(outputs, targets)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """预测（用于推理）"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def get_parameter_count(self) -> Dict[str, int]:
        """获取参数数量"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        backbone_params = sum(p.numel() for n, p in self.model.named_parameters() if not n.startswith('fc'))
        classifier_params = sum(p.numel() for n, p in self.model.named_parameters() if n.startswith('fc'))

        return {
            'total': total_params,
            'trainable': trainable_params,
            'backbone': backbone_params,
            'classifier': classifier_params
        }

    @classmethod
    def from_pretrained(cls, num_classes: int = 1000, weights: str = 'DEFAULT', **kwargs):
        """创建预训练的 ResNet-50 模型"""
        config = {
            'num_classes': num_classes,
            'pretrained': True,
            'weights': weights,
            **kwargs
        }
        return cls(config)

    def freeze_backbone(self):
        """冻结主干网络参数"""
        for name, param in self.model.named_parameters():
            if not name.startswith('fc'):
                param.requires_grad = False
        self.logger.info("主干网络已冻结")

    def unfreeze_backbone(self):
        """解冻主干网络参数"""
        for param in self.model.parameters():
            param.requires_grad = True
        self.logger.info("主干网络已解冻")

    def get_feature_extractor(self):
        """获取特征提取器（不包含分类层）"""
        feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        return feature_extractor
