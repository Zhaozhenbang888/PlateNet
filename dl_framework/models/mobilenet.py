import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
import timm
import os

try:
    from .base_model import BaseModel
    from .registry import ModelRegistry
except ImportError:
    # 在直接运行此文件时的导入方式
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from dl_framework.models.base_model import BaseModel
    from dl_framework.models.registry import ModelRegistry


class ConvBNReLU(nn.Module):
    """卷积+批归一化+ReLU激活的基本模块"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=groups, 
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InvertedResidual(nn.Module):
    """MobileNetV2的倒置残差模块"""
    
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # 扩展层
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1, padding=0))
        
        layers.extend([
            # 深度可分离卷积
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # 投影层
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

@ModelRegistry.register('mobilenet_v2')
class MobileNetV2(BaseModel):
    """MobileNetV2模型
    
    使用倒置残差结构和线性瓶颈的轻量级CNN模型
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化
        
        Args:
            config: 模型配置
                - num_classes: 类别数，默认为1000
                - width_mult: 宽度乘数，用于调整网络宽度，默认为1.0
                - input_size: 输入图像大小，默认为224
                - dropout_rate: Dropout率，默认为0.2
                - pretrained: 是否使用预训练模型，默认为True
                - freeze_backbone: 是否冻结主干网络，默认为False
        """
        super(MobileNetV2, self).__init__(config)
        
        self.num_classes = config.get('num_classes', 1000)
        self.width_mult = config.get('width_mult', 1.0)
        self.input_size = config.get('input_size', 224)
        self.dropout_rate = config.get('dropout_rate', 0.2)
        self.pretrained = config.get('pretrained', True)
        self.freeze_backbone = config.get('freeze_backbone', False)
        
        # 使用timm库加载预训练的MobileNetV2模型
        model_name = f'mobilenetv2_{int(self.width_mult * 100)}'
        self.model = timm.create_model(
            model_name, 
            pretrained=self.pretrained,
            num_classes=self.num_classes,
            drop_rate=self.dropout_rate
        )
            
        # 如果启用了冻结主干网络，则冻结除分类头以外的所有参数
        if self.freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """冻结模型的主干网络，只保留分类头可训练"""
        print(f"冻结 MobileNetV2 的主干网络，只保留分类头可训练")
        
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
    
    def is_backbone_frozen(self):
        """检查主干网络是否已被冻结"""
        # 检查第一层参数是否被冻结
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                return not param.requires_grad
        return False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入图像数据，形状为 [batch_size, channels, height, width]
            
        Returns:
            分类结果，形状为 [batch_size, num_classes]
        """
        return self.model(x)
    
    def get_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算损失
        
        Args:
            outputs: 模型输出
            targets: 目标值
            
        Returns:
            损失值
        """
        return F.cross_entropy(outputs, targets)
