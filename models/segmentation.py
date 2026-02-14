import torch
import torch.nn as nn
from transformers import ViTMAEForPreTraining


class ViTMAEDecoder(nn.Module):
    """
    纯 forward 的 ViT-MAE 解码器（训练专用）
    输入: (B, 196, 768) 隐向量（ViT 编码器输出）
    输出: (B, 1, 224, 224) 重建灰度图像（值域 [-1, 1]，符合 MAE 训练协议）
    
    ✅ 特性:
      - 输出单通道灰度图（非简单取 R 通道）
      - 使用标准亮度公式 Y = 0.299*R + 0.587*G + 0.114*B
      - 值域严格保持 [-1, 1]（与 MAE 训练协议一致）
      - 支持梯度反向传播（所有参数可训练）
      - 自动 device/buffer 管理（无需手动 .to(device)）
      - 冻结选项（freeze_decoder=True）
    """

    def __init__(
        self,
        pretrained_model_name: str = "facebook/vit-mae-base",
        freeze_decoder: bool = False,
        patch_size: int = 16,
        img_size: int = 224,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_size = img_size // patch_size  # 14 for 224x224

        # 加载预训练 decoder（仅 decoder 部分）
        model = ViTMAEForPreTraining.from_pretrained(pretrained_model_name)
        self.decoder = model.decoder

        # 冻结 decoder 参数（可选）
        if freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False

        # 注册 ids_restore 为 buffer（不参与梯度，但随 device 自动迁移）
        ids_restore = torch.arange(self.grid_size**2)  # [196]
        self.register_buffer("ids_restore", ids_restore.unsqueeze(0))  # [1, 196]

        # ✅ 新增：注册灰度转换权重（ITU-R BT.601 标准）
        # 权重: [0.299, 0.587, 0.114] → 形状 [3, 1, 1] 便于广播
        self.register_buffer(
            "rgb_weights", 
            torch.tensor([0.299, 0.587, 0.114], dtype=torch.float).view(3, 1, 1)
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: torch.Tensor, shape=[B, 196, 768]
                    ViT 编码器输出的隐向量（已归一化）
        
        Returns:
            recon_gray: torch.Tensor, shape=[B, 1, 224, 224]
                        重建灰度图像，值域 [-1, 1]
        """
        B = latent.shape[0]

        # 扩展 ids_restore 到 batch 维度: [1,196] → [B,196]
        ids_restore_batch = self.ids_restore.expand(B, -1)

        # 执行解码
        decoder_output = self.decoder(
            hidden_states=latent,
            ids_restore=ids_restore_batch,
        )
        pred = decoder_output.logits  # [B, 196, 768]

        # unpatchify → [B, 3, 224, 224] (RGB)
        h = w = self.grid_size
        pred = pred.reshape(B, h, w, self.patch_size, self.patch_size, 3)
        pred = pred.permute(0, 5, 1, 3, 2, 4)  # [B, 3, h, p, w, p]
        recon_rgb = pred.reshape(B, 3, self.img_size, self.img_size)

        # ✅ 核心：转为 [B, 1, 224, 224] 灰度图（可微分、设备自适应）
        # 使用标准亮度公式 Y = 0.299*R + 0.587*G + 0.114*B
        recon_gray = torch.sum(recon_rgb * self.rgb_weights, dim=1, keepdim=True)
        
        return recon_gray  # [B, 1, 224, 224], 值域 [-1, 1]


# ✅ 使用示例（训练场景）
if __name__ == "__main__":
    # 初始化（可训练）
    decoder = ViTMAEDecoder(freeze_decoder=False)

    # 模拟您的编码器输出（带梯度）
    latent = torch.randn(4, 196, 768, requires_grad=True)

    # 前向传播（训练模式）
    recon_gray = decoder(latent)  # [4, 1, 224, 224], requires_grad=True
    print(f"✅ forward 输出形状: {recon_gray.shape}")
    print(f"✅ 是否可求梯度: {recon_gray.requires_grad}")
    print(f"✅ 值域范围: [{recon_gray.min():.3f}, {recon_gray.max():.3f}]")

    # 示例损失 + 反向传播
    target_gray = torch.randn(4, 1, 224, 224)  # 单通道目标
    loss = torch.nn.functional.mse_loss(recon_gray, target_gray)
    loss.backward()
    print(f"✅ latent.grad 形状: {latent.grad.shape}")
