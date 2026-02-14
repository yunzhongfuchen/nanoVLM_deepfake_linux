#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立测试脚本：在 SID (深伪认证) 测试集上评估模型性能
Usage:
    python test_auth.py --checkpoint_path ./checkpoints/final_model
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

# 避免 tokenizer 多进程警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ===== 核心依赖（请根据实际项目结构调整导入路径）=====
from data.datasets import SIDataset
from data.collators import VQACollator
from data.processors import get_image_processor, get_tokenizer
from models.vision_language_model import VisionLanguageModel
import models.config as config


def test_auth_dataset(model, tokenizer, test_loader, device):
    """
    在认证数据集（SID 验证集）上评估模型
    逻辑：从生成文本中提取关键标签（real/tampered/full_synthetic）进行匹配
    """
    model.eval()
    total, correct = 0, 0
    print_count = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device, non_blocking=True)  # 修正键名：image (非 images)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            labels_cls = batch.get("labels_cls", None)
            if labels_cls is not None:
                labels_cls = labels_cls.to(device, non_blocking=True)

            batch_size = input_ids.size(0)
            
            # 1) 从 labels 提取标准答案文本（仅保留答案部分）
            gt_answers = []
            for i in range(batch_size):
                label_ids = labels[i]
                ans_ids = label_ids[label_ids != -100]
                if ans_ids.numel() == 0:
                    gt_answers.append("")
                else:
                    gt = tokenizer.decode(ans_ids, skip_special_tokens=False).strip().lower()
                    gt_answers.append(gt)

            # 2) 模型生成答案
            # 注意：假设 model.generate 返回 (gen_ids, cls_pred)；若模型无分类头，需调整
            try:
                gen_output = model.generate(input_ids, images, attention_mask)
                if isinstance(gen_output, tuple) and len(gen_output) == 2:
                    gen_ids, cls_pred = gen_output
                else:
                    gen_ids, cls_pred = gen_output, None
            except Exception as e:
                print(f"[WARNING] Model generate failed: {e}. Using fallback.")
                gen_ids = model.generate(input_ids, images, attention_mask)
                cls_pred = None

            pred_answers = tokenizer.batch_decode(gen_ids, skip_special_tokens=False)
            pred_answers = [p.strip().lower() for p in pred_answers]

            # 3) 调试信息（仅前3个样本）
            if print_count < 3 and batch_size > 0:
                print("=" * 60)
                print(f"Sample #{print_count + 1}")
                print(f"  GT Text  : '{gt_answers[0]}'")
                print(f"  Pred Text: '{pred_answers[0]}'")
                if cls_pred is not None and labels_cls is not None:
                    print(f"  GT Label : {labels_cls[0].item() if hasattr(labels_cls[0], 'item') else labels_cls[0]}")
                    print(f"  Pred Cls : {cls_pred[0].item() if hasattr(cls_pred[0], 'item') else cls_pred[0]}")
                print_count += 1

            # 4) 答案标准化与匹配
            for pred, gt in zip(pred_answers, gt_answers):
                # 标准化 GT 标签
                if "real" in gt:
                    gt_label = "real"
                elif "tampered" in gt or "manipulated" in gt:
                    gt_label = "tampered"
                elif "full" in gt and "synthetic" in gt:
                    gt_label = "full_synthetic"
                else:
                    gt_label = gt

                # 标准化预测标签
                if "real" in pred:
                    pred_label = "real"
                elif "tampered" in pred or "manipulated" in pred:
                    pred_label = "tampered"
                elif "full" in pred and "synthetic" in pred:
                    pred_label = "full_synthetic"
                else:
                    pred_label = pred

                if gt_label == pred_label:
                    correct += 1
                total += 1

    model.train()  # 恢复训练模式（若后续需继续训练）
    accuracy = correct / total if total > 0 else 0.0
    print("\n" + "=" * 60)
    print(f"✅ 认证数据集评估完成 | 准确率: {accuracy:.4f} ({correct}/{total})")
    print("=" * 60)
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="SID 深伪认证数据集独立测试脚本")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="模型检查点路径 (需包含 model.safetensors 或 pytorch_model.bin)")
    parser.add_argument("--dataset_path", type=str, default="Lin-Chen/SID",
                        help="SID 数据集路径 (HuggingFace 格式)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="测试批次大小")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader 工作进程数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="运行设备 (cuda/cpu)")
    args = parser.parse_args()

    print(f"[INFO] 使用设备: {args.device}")
    device = torch.device(args.device)

    # === 1. 加载配置与处理器 ===
    vlm_cfg = config.VLMConfig()
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)
    image_processor = get_image_processor(vlm_cfg.vit_img_size)

    # === 2. 构建测试数据集与 DataLoader ===
    print(f"[INFO] 加载 SID 测试集 (验证集): {args.dataset_path}")
    sid_dataset = load_dataset(args.dataset_path)
    test_dataset = SIDataset(sid_dataset["validation"], tokenizer, image_processor)
    
    collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    print(f"[INFO] 测试集样本数: {len(test_dataset)} | Batch Size: {args.batch_size}")

    # === 3. 加载模型 ===
    print(f"[INFO] 从 {args.checkpoint_path} 加载模型...")
    model = VisionLanguageModel.from_pretrained(args.checkpoint_path)
    model.decoder.resize_token_embeddings(len(tokenizer))  # 适配 tokenizer 词表
    model.to(device)
    model.eval()
    print(f"[INFO] 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # === 4. 执行评估 ===
    accuracy = test_auth_dataset(model, tokenizer, test_loader, device)
    print(f"\n🎯 最终认证准确率: {accuracy:.4%}\n")

    # 可选：保存结果到文件
    with open("auth_test_result.txt", "w") as f:
        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write(f"Dataset: {args.dataset_path}\n")
        f.write(f"Accuracy: {accuracy:.6f}\n")
    print("[INFO] 结果已保存至 auth_test_result.txt")


if __name__ == "__main__":
    main()
