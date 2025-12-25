from transformers import AutoTokenizer
import torchvision.transforms as transforms

TOKENIZERS_CACHE = {}

def get_tokenizer(name):
    if name not in TOKENIZERS_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        # 显式检查每个 token 是否已存在
        new_tokens = ["<CLS>", "<SEG>"]
        tokens_to_add = [t for t in new_tokens if t not in tokenizer.get_vocab()]

        if tokens_to_add:
            num_added = tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
            print(f"✅ 添加了 {num_added} 个新特殊 token: {tokens_to_add}")
        else:
            print("🟢 所需 token 已存在，无需添加")

        print(f"最终词汇表大小: {len(tokenizer)}")
        TOKENIZERS_CACHE[name] = tokenizer
    return TOKENIZERS_CACHE[name]

def get_image_processor(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

def get_mask_processor(img_size):
    """用于二值掩码（tampering mask）的预处理器"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),  # [0, 255] -> [0, 1], shape (1, H, W)
        transforms.Lambda(lambda x: (x > 0.5).float())  # 强制二值化为 0.0 / 1.0
    ])