from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers import Tokenizer
import os

data_size = int(os.environ.get('DATA_SIZE', 50000))
vocab_size = int(os.environ.get('VOCAB_SIZE', 40000))
print(f"data_size: {data_size}")
print(f"vocab_size: {vocab_size}")

input_file = f"data/gigaword_input_{data_size}.txt"
summary_file = f"data/gigaword_summary_{data_size}.txt"
texts = []

with open(input_file, "r") as f:
    for line in f:
        texts.append(line.strip())

with open(summary_file, "r") as f:
    for line in f:
        texts.append(line.strip())

min_frequency = 2   # 最小词频

# 初始化 tokenizer
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()

# 配置训练器
trainer = BpeTrainer(
    vocab_size=vocab_size,
    min_frequency=min_frequency,
    special_tokens=["<pad>", "<bos>", "<eos>", "<unk>", "<num>"]
)

# 训练模型
tokenizer.train_from_iterator(texts, trainer)

# 设置特殊 token 的属性
tokenizer.pad_token = "<pad>"
tokenizer.pad_token_id = tokenizer.token_to_id("<pad>")

tokenizer.bos_token = "<bos>"
tokenizer.bos_token_id = tokenizer.token_to_id("<bos>")

tokenizer.eos_token = "<eos>"
tokenizer.eos_token_id = tokenizer.token_to_id("<eos>")

tokenizer.unk_token = "<unk>"
tokenizer.unk_token_id = tokenizer.token_to_id("<unk>")

# 保存模型和词汇表
model_path = f"models/bpe_model_{data_size}.json"
tokenizer.save(model_path)
print(f"BPE 模型已保存到 {model_path}")

# 获取词汇表
vocab = tokenizer.get_vocab()
print(f"词汇表大小: {len(vocab)}")

# 保存词汇表
vocab_path = f"models/bpe_vocab_{data_size}.txt"
with open(vocab_path, "w", encoding="utf-8") as f:
    for token, _ in sorted(vocab.items(), key=lambda x: x[1]):
        f.write(f"{token}\n")

print(f"词汇表已保存到 {vocab_path}")

# 测试 tokenizer
test_text = texts[0]
encoded = tokenizer.encode(test_text)
print(f"原文: {test_text}")
print(f"分词结果: {encoded.tokens}")
