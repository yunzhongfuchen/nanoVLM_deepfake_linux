import torch
from PIL import Image
from torch.utils.data import Dataset

import models.config as cfg


class VQADataset(Dataset):  # Visual Question Answering Dataset
    def __init__(self, dataset, tokenizer, image_processor):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Handle image (it's a list)
        image_data = item['images']
        if isinstance(image_data, list) and len(image_data) > 0:
            image = image_data[0]
        else:
            image = image_data

        # Now process the image
        if isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            processed_image = self.image_processor(image)
        else:
            print(f"Error processing image at index {idx}")
            # Create empty tensor with right dimensions as fallback
            processed_image = torch.zeros(
                3, cfg.VLMConfig.vit_img_size, cfg.VLMConfig.vit_img_size)

        # Process text (also a list)
        text_data = item['texts']
        if isinstance(text_data, list) and len(text_data) > 0:
            text = text_data[0]
        else:
            text = text_data

        question = text['user']
        # Add EOS token to the answer to train model to predict it, enabling correct stopping during generation
        answer = text['assistant'] + self.tokenizer.eos_token

        formatted_text = f"Question: {question} Answer:"

        return {
            "image": processed_image,
            "text_data": formatted_text,
            "answer": answer
        }


class MMStarDataset(Dataset):  # https://huggingface.co/datasets/Lin-Chen/MMStar
    def __init__(self, dataset, tokenizer, image_processor):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        image = item['image']
            
        # Now process the image
        if isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            processed_image = self.image_processor(image)
        else:
            print(f"Error processing image at index {idx}")
            # Create empty tensor with right dimensions as fallback
            processed_image = torch.zeros(3, cfg.VLMConfig.vit_img_size, cfg.VLMConfig.vit_img_size)
        
        question = item['question']
        answer = item['answer'] + self.tokenizer.eos_token # Add EOS token to the answer to train model to predict it, enabling correct stopping during generation
        
        formatted_text = f"Question: {question} \nAnswer only with the letter! \nAnswer:"
        
        return {
            "image": processed_image,
            "text_data": formatted_text,
            "answer": answer
        }
   
    
import numpy as np

class SIDataset(Dataset):
    """Synthetic Image Detection Dataset - 分辨真实/合成/篡改图像"""
    def __init__(self, dataset, tokenizer, image_processor, mask_processor=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mask_processor = mask_processor
        self.label_map = {
            0: "<CLS> this is real image.",
            1: "<CLS> this is full synthetic image.",
            2: "<CLS> this is tampered image. <SEG>"
        }
        self.question_prompt = "Question: Is this image real, full synthetic or tampered? Answer:"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # === 处理主图像 ===
        image = item['image']
        if not isinstance(image, Image.Image):
            warnings.warn(f"Invalid image type at index {idx}, expected PIL.Image. Using zero tensor.")
            processed_image = torch.zeros(3, cfg.VLMConfig.vit_img_size, cfg.VLMConfig.vit_img_size)
        else:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            processed_image = self.image_processor(image)

        # === 处理标签和答案 ===
        label = item['label']
        if label not in self.label_map:
            raise ValueError(f"Invalid label {label} at index {idx}. Must be 0, 1, or 2.")
        answer = self.label_map[label] + self.tokenizer.eos_token * 3

        # === 处理掩码（仅 tampered image 有，但数据集可能统一提供）===
 # === 处理掩码（仅 tampered image 有，但数据集可能统一提供）===
        mask = item.get('mask', None)
        processed_mask = None
        if mask is not None:
            if not isinstance(mask, Image.Image):
                warnings.warn(f"Invalid mask type at index {idx}, expected PIL.Image. Using zero mask.")
                processed_mask = torch.zeros(1, cfg.VLMConfig.vit_img_size, cfg.VLMConfig.vit_img_size)
            else:
                if mask.mode != 'L':
                    mask = mask.convert('L')
                if self.mask_processor is not None:
                    processed_mask = self.mask_processor(mask)
                else:
                    # 如果未提供 mask_processor，尝试用 image_processor（不推荐）
                    try:
                        processed_mask = self.image_processor(mask)  # 可能出错或产生非二值结果
                    except Exception as e:
                        warnings.warn(f"Failed to process mask at index {idx}: {e}. Using zero mask.")
                        processed_mask = torch.zeros(1, cfg.VLMConfig.vit_img_size, cfg.VLMConfig.vit_img_size)
        else:
            # ✅ 自动生成全零掩码（统一输出结构）
            processed_mask = torch.zeros(1, cfg.VLMConfig.vit_img_size, cfg.VLMConfig.vit_img_size)
        return {
            "image": processed_image,      # (3, H, W)
            "text_data": self.question_prompt,
            "answer": answer,              # str with EOS tokens
            "mask": processed_mask,        # (1, H, W) float tensor of 0.0/1.0, or None
            "label": label                 # int: 0, 1, or 2
        }

import os
from pathlib import Path
class AuthFolderDataset(Dataset):
    """
    从一个大目录下的三个子文件夹(real, tampered, full_synthetic)读取图片，
    并把类别转成文本答案，例如:
        real          -> "The image is real"
        tampered      -> "The image is tampered"
        full_synthetic-> "The image is full synthetic"
    """

    def __init__(self, root_dir, tokenizer, image_processor):
        self.root_dir = Path(root_dir)
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        # 文件夹名 -> 文本答案
        self.class_to_answer = {
            "real": "<CLS> The image is real",
            "tampered": "<CLS> The image is tampered <SEG>",
            "full_synthetic": "<CLS> The image is full synthetic",
        }

        # 新增：文件夹名 -> 分类标签（按你的要求）
        self.class_to_label = {
            "real": 0,
            "tampered": 2,
            "full_synthetic": 1,
        }

        self.class_names = list(self.class_to_answer.keys())

        # 收集样本 (path, class_name)
        self.samples = []
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for cls in self.class_names:
            cls_dir = self.root_dir / cls
            if not cls_dir.is_dir():
                print(f"[WARN] 子目录不存在: {cls_dir}")
                continue

            for fname in os.listdir(cls_dir):
                path = cls_dir / fname
                if path.suffix.lower() not in exts:
                    continue
                self.samples.append((path, cls))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"在 {self.root_dir} 下没有找到任何图片（real/tampered/full_synthetic）"
            )

        print(f"[INFO] AuthFolderDataset: 共 {len(self.samples)} 张图片")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cls_name = self.samples[idx]

        # --- 读图 ---
        try:
            image = Image.open(img_path)
        except Exception as e:
            print(f"[ERROR] 打开图片失败: {img_path}, error: {e}")
            processed_image = torch.zeros(
                3, cfg.VLMConfig.vit_img_size, cfg.VLMConfig.vit_img_size
            )
        else:
            if image.mode != "RGB":
                image = image.convert("RGB")
            processed_image = self.image_processor(image)

        # --- 构造 QA 文本 ---
        question = "Classify the authenticity of this image as real, tampered, or full synthetic."
        answer_text = self.class_to_answer[cls_name]
        answer = answer_text + self.tokenizer.eos_token * 3
        formatted_text = f"Question: {question} Answer:"

        # --- 获取分类标签 ---
        cls_label = self.class_to_label[cls_name]
        processed_mask = torch.zeros(1, cfg.VLMConfig.vit_img_size, cfg.VLMConfig.vit_img_size)
        return {
            "image": processed_image,
            "text_data": formatted_text,
            "answer": answer,
            "mask": processed_mask,
            "label": cls_label,  # 👈 新增字段
        }
