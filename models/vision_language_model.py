import json
import os
import tempfile
from dataclasses import asdict
from typing import Optional


from models.vision_transformer import ViT
from models.language_model import LanguageModel
from models.modality_projector import ModalityProjector
from models.segmentation import ViTMAEDecoder
from models.config import VLMConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model, load_file
from data.processors import get_image_processor, get_tokenizer

class VisionLanguageModel(nn.Module):
    def __init__(self, cfg: VLMConfig, load_backbone=True):
        super().__init__()
        self.cfg = cfg
        if load_backbone:
            print("Loading from backbone weights")
            self.vision_encoder = ViT.from_pretrained(cfg)
            self.decoder = LanguageModel.from_pretrained(cfg)
        else:
            self.vision_encoder = ViT(cfg)
            self.decoder = LanguageModel(cfg)
        self.MP = ModalityProjector(cfg)
        # === 分类头：从 [CLS] 向量预测类别 ===
        # 我们使用一个小型 MLP：hidden_size -> hidden_size -> num_classes
        self.tokenizer = get_tokenizer(cfg.lm_tokenizer)
        self.cls_token_id = self.tokenizer.convert_tokens_to_ids("<CLS>")
        self.hidden_size = cfg.lm_hidden_dim  # 用于分类头
        self.num_classes = 3  # 三分类任务
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),  # 可调
            nn.Linear(self.hidden_size, self.num_classes)
        )
        # 视觉解码器
        self.vision_decoder = ViTMAEDecoder(freeze_decoder=False)
        self.load_backbone = load_backbone

    def forward(self, input_ids, image, attention_mask, targets, targets_cls, targets_mask):
        image_embd = self.vision_encoder(image)
        # 🔥 新增：视觉重建（在 MP 之前！）
        mask_pred = self.vision_decoder(image_embd)  # [B, 3, 224, 224]
        image_loss = None
        if targets_mask is not None:
            image_loss = F.mse_loss(mask_pred, targets_mask)  # 计算重建损失

            
        image_embd = self.MP(image_embd)

        token_embd = self.decoder.token_embedding(input_ids)

        combined_embd = torch.cat((image_embd, token_embd), dim=1) # Concatenate image embeddings to token embeddings
        
        # Adjust attention mask to account for image tokens
        if attention_mask is not None:
            # Create mask of 1s for image tokens (all image tokens should be attended to)
            batch_size = image_embd.size(0)
            img_seq_len = image_embd.size(1)
            image_attention_mask = torch.ones((batch_size, img_seq_len), device=attention_mask.device, dtype=attention_mask.dtype)
            
            # Combine image and token attention masks
            attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)


        hidden_states = self.decoder(combined_embd, attention_mask)  # [B, N_img+T, D]

        # cls_hidden = self._extract_token_hidden_states(
        #     hidden_states=hidden_states,
        #     token_ids=input_ids,
        #     target_token_id=self.cls_token_id,
        #     img_seq_len=image_embd.size(1),
        #     fallback_to_img_last=True
        # )
        cls_hidden = self._extract_token_hidden_states(
            hidden_states=hidden_states,
            absolute_position=64,
            full_token_ids=input_ids,      # ← 传入 input_ids 用于解码
            debug=False
        )
        class_logits = self.classifier(cls_hidden).squeeze(1)  # [B, 3]

        cls_loss = None
        if targets_cls is not None:
            cls_loss = F.cross_entropy(class_logits, targets_cls)

        # === 现在可以安全地将 logits 改写为语言建模输出 ===
        loss = None
        if targets is not None:
            # Only use the token part of the logits for loss computation
            logits = self.decoder.head(hidden_states)
            logits = logits[:, image_embd.size(1):, :]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)

        # === 合并总损失 ===
        # total_loss = None
        # if loss is not None and cls_loss is not None:
        #     total_loss = loss + cls_loss
        # elif loss is not None:
        #     total_loss = loss
        # elif cls_loss is not None:
        #     total_loss = cls_loss
        total_loss = None
        if loss is not None:
            total_loss = loss
        if cls_loss is not None:
            total_loss += cls_loss
        if image_loss is not None:
            total_loss += image_loss  # ← 从重建损失开始

        # === 返回三项结果 ===
        return logits, total_loss, class_logits, mask_pred


    @torch.no_grad()
    def generate(self, input_ids, image, attention_mask=None, max_new_tokens=20):
        # Process image through vision encoder and projection
        image_embd = self.vision_encoder(image)
        mask_pred = self.vision_decoder(image_embd)
        image_embd = self.MP(image_embd)
        
        # Embed initial tokens
        token_embd = self.decoder.token_embedding(input_ids)
        
        # Concatenate image embeddings with token embeddings
        combined_embd = torch.cat((image_embd, token_embd), dim=1)

        batch_size = image_embd.size(0)
        img_seq_len = image_embd.size(1)
        # Adjust attention mask to account for image tokens
        if attention_mask is not None:
            # Create mask of 1s for image tokens (all image tokens should be attended to)
            image_attention_mask = torch.ones((batch_size, img_seq_len), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)
    

        # Generate from combined embeddings using the decoder
        # We need to use the decoder's forward function and not its generate method
        # because we want to keep track of the image prefix
        outputs = combined_embd
        generated_tokens = torch.zeros((batch_size, max_new_tokens), device=input_ids.device, dtype=input_ids.dtype)
        
        #Note: Here you could implement improvements like e.g. KV caching
        for i in range(max_new_tokens):
            model_out = self.decoder(outputs, attention_mask)
            
            # Get predictions for the last token only (normally this is the embedding, not the logits)
            last_token_logits = model_out[:, -1, :]
            
            # Apply head to get logits (if model is in embedding mode)
            if not self.decoder.lm_use_tokens:
                last_token_logits = self.decoder.head(last_token_logits)

            probs = torch.softmax(last_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
                
            generated_tokens[:, i] = next_token.squeeze(-1)
            
            # Convert to embedding and append
            next_embd = self.decoder.token_embedding(next_token)
            outputs = torch.cat((outputs, next_embd), dim=1)

            if attention_mask is not None:
                attention_mask = torch.cat((attention_mask, torch.ones((batch_size, 1), device=attention_mask.device)), dim=1)
        
        # 再次前向传播，获取最终的 hidden states（包含所有生成 token 的上下文表示）
        full_token_ids = torch.cat([input_ids, generated_tokens], dim=1)  # [B, prompt_len + max_new_tokens]

        final_hidden_states = self.decoder(outputs, attention_mask)

        # === 使用统一函数提取 [CLS] ===
        # cls_hidden = self._extract_token_hidden_states(
        #     hidden_states=final_hidden_states,
        #     token_ids=full_token_ids,
        #     target_token_id=self.cls_token_id,
        #     img_seq_len=image_embd.size(1),
        #     fallback_to_img_last=True
        # )
        cls_hidden = self._extract_token_hidden_states(
            hidden_states=final_hidden_states,
            absolute_position=64,
            full_token_ids=full_token_ids, # ← 传入完整文本 tokens
            debug=False
)
        class_logits = self.classifier(cls_hidden)  # [B, num_classes]
        cls_pred = class_logits.argmax(dim=-1)
        # === ✅ 修改返回值 ===
        return generated_tokens, cls_pred, mask_pred

    # def _extract_token_hidden_states(
    #     self,
    #     hidden_states,          # [B, L, D] - 完整序列的 hidden states
    #     token_ids,              # [B, T] - 原始 token ID 序列（仅文本部分）
    #     target_token_id,        # int - 要查找的 token ID（如 cls_token_id）
    #     img_seq_len,            # int - 图像 token 数量
    #     fallback_to_img_last=True  # bool - 未找到时是否回退到图像最后一个 token
    # ):
    #     """
    #     从完整 hidden_states 中提取第一个 target_token_id 对应的隐藏状态。
        
    #     Args:
    #         hidden_states: [B, total_seq_len, D]
    #         token_ids: [B, text_seq_len] —— 注意：只包含文本 token（不含图像）
    #         target_token_id: 要查找的 token ID（如 self.cls_token_id）
    #         img_seq_len: 图像部分的长度
    #         fallback_to_img_last: 若未找到，是否回退到图像最后一个 token（位置 img_seq_len - 1）
        
    #     Returns:
    #         extracted_hidden: [B, D] —— 每个样本提取出的隐藏状态
    #     """
    #     batch_size = hidden_states.size(0)
    #     device = hidden_states.device

    #     # 在文本 token 中查找目标 token
    #     cls_mask = (token_ids == target_token_id)  # [B, T]
    #     has_target = cls_mask.any(dim=1)           # [B]
    #     first_pos_in_text = cls_mask.long().argmax(dim=1)  # [B]

    #     # 计算绝对位置（在完整序列中的索引）
    #     if fallback_to_img_last:
    #         fallback_pos = img_seq_len - 1  # 图像最后一个 token
    #     else:
    #         fallback_pos = 0  # 或抛出错误，根据需求

    #     abs_positions = torch.where(
    #         has_target,
    #         img_seq_len + first_pos_in_text,  # 文本部分从 img_seq_len 开始
    #         fallback_pos
    #     )  # [B]

    #     # 向量化提取
    #     batch_indices = torch.arange(batch_size, device=device)
    #     extracted_hidden = hidden_states[batch_indices, abs_positions, :]  # [B, D]
        
    def _extract_token_hidden_states(
        self,
        hidden_states,          # [B, L, D]
        absolute_position=64,   # 固定位置
        full_token_ids=None,    # [B, L_text] - 完整文本 token IDs（用于调试，可选）
        debug=False             # 是否打印
    ):
        """
        从 hidden_states 中提取固定绝对位置的隐藏状态。
        
        Args:
            hidden_states: [B, total_seq_len, D]
            absolute_position: int
            full_token_ids: [B, text_seq_len], 仅用于调试解码（不含图像 tokens）
            debug: bool
        """
        batch_size = hidden_states.size(0)
        total_seq_len = hidden_states.size(1)
        device = hidden_states.device

        if absolute_position >= total_seq_len:
            raise IndexError(f"Position {absolute_position} out of range (seq_len={total_seq_len})")

        if debug:
            print(f"\n=== Extracting hidden state at absolute position: {absolute_position} ===")
            if full_token_ids is not None:
                img_seq_len = total_seq_len - full_token_ids.size(1)  # 推断图像长度
                text_pos = absolute_position - img_seq_len
                if text_pos >= 0 and text_pos < full_token_ids.size(1):
                    for b in range(min(batch_size, 2)):  # 只打印前2个样本
                        token_id = full_token_ids[b, text_pos].item()
                        decoded = "N/A"
                        if hasattr(self, 'tokenizer'):
                            try:
                                decoded = self.tokenizer.decode([token_id], skip_special_tokens=False)
                            except:
                                pass
                        print(f"  Sample {b}: token_id={token_id}, decoded='{decoded}' (text pos={text_pos})")
                else:
                    print(f"  Warning: absolute_position {absolute_position} is within image tokens (img_seq_len={img_seq_len})")
            else:
                print("  (full_token_ids not provided, skipping token decoding)")
            print("==================================================\n")

        batch_indices = torch.arange(batch_size, device=device)
        extracted_hidden = hidden_states[batch_indices, absolute_position, :]
        return extracted_hidden


    @classmethod
    def from_pretrained(
        cls, repo_id_or_path: str, *, revision: Optional[str] = None
    ) -> "VisionLanguageModel":
        """
        Load a VisionLanguageModel from a local directory or a repo on the Hugging Face Hub.

        Args:
            repo_id_or_path (str): The path to the local directory or the Hugging Face Hub repo ID.

        Returns:
            VisionLanguageModel: The loaded model.
        """
        # If local folder exists => load from there
        if os.path.exists(repo_id_or_path):
            config_path = os.path.join(repo_id_or_path, "config.json")
            weights_path = os.path.join(repo_id_or_path, "model.safetensors")

            if not os.path.exists(config_path):
                raise ValueError(
                    f"Config file not found at {config_path}. Please provide a valid path."
                )
            if not os.path.exists(weights_path):
                raise ValueError(
                    f"Weights file not found at {weights_path}. Please provide a valid path."
                )
        # Otherwise, assume it's a Hugging Face Hub repo
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="config.json", revision=revision
            )
            weights_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="model.safetensors", revision=revision
            )

        # Load config
        with open(config_path, "r") as f:
            cfg = VLMConfig(**json.load(f))

        # Initialize model without loading the backbone
        model = cls(cfg, load_backbone=False)

        # Load safetensors weights
        # === 修改此处：替换原 load_model(model, weights_path) ==

        # 获取模型设备（安全处理无参数情况）
        device = next(model.parameters()).device if next(model.parameters(), None) is not None else torch.device("cpu")
        
        # 加载并过滤：仅保留键存在且形状匹配的参数
        state_dict = load_file(weights_path, device=str(device))
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if k in model.state_dict() and v.shape == model.state_dict()[k].shape
        }
        model.load_state_dict(filtered_state_dict, strict=False)  # strict=False 允许模型有额外参数
        print("✅ Model loaded")
        # === 修改结束 ===

        # Done!
        return model

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the model and configuration to a directory.

        Args:
            save_directory (str): The directory to save the model and config.
        """
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write(json.dumps(asdict(self.cfg), indent=4))

        # Save weights as safetensors
        save_model(self, os.path.join(save_directory, "model.safetensors"))

    def push_to_hub(self, repo_id: str, private: bool = False) -> None:
        """
        Push the model and configuration to the Hugging Face Hub.

        Args:
            repo_id (str): The repo ID on the Hugging Face Hub.
        """
        from huggingface_hub import create_repo, upload_folder

        # Create repo
        repo_url = create_repo(repo_id=repo_id, private=private, exist_ok=True)
        repo_id = repo_url.repo_id
        print("Created repo: ", repo_url)

        with tempfile.TemporaryDirectory() as save_path:
            # Save to tmp directory
            self.save_pretrained(save_path)

            # Save model card
            with open(os.path.join(save_path, "README.md"), "w") as f:
                f.write(MODEL_CARD_TEMPLATE.format(repo_id=repo_id))

            # Upload
            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=save_path,
                commit_message="Upload nanoVLM using push_to_hub",
            )


MODEL_CARD_TEMPLATE = """
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
library_name: nanovlm
license: mit
pipeline_tag: image-text-to-text
tags:
  - vision-language
  - multimodal
  - research
---

**nanoVLM** is a minimal and lightweight Vision-Language Model (VLM) designed for efficient training and experimentation. Built using pure PyTorch, the entire model architecture and training logic fits within ~750 lines of code. It combines a ViT-based image encoder (SigLIP-B/16-224-85M) with a lightweight causal language model (SmolLM2-135M), resulting in a compact 222M parameter model.

For more information, check out the base model on https://huggingface.co/lusxvr/nanoVLM-222M.

**Usage:**

Clone the nanoVLM repository: https://github.com/huggingface/nanoVLM.
Follow the install instructions and run the following code:

```python
from models.vision_language_model import VisionLanguageModel

model = VisionLanguageModel.from_pretrained("{repo_id}")
```
"""
