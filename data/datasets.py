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