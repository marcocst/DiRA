import json

# 读取JSON文件
with open('generated_predictions_2025_8_6_111.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# 计算准确率
correct = 0
total = len(data)

for item in data:
    if item['predict'].strip().lower()[-4:] == item['label'].strip().lower()[-4:]:
        correct += 1

accuracy = correct / total * 100

print(f"准确率: {accuracy:.2f}%")
print(f"正确数量: {correct}/{total}")