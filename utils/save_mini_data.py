import pandas as pd

# 定义输入输出文件路径
# input_file_path = "data/pt/accommodation_catering_hotel/chinese/high/rank_00000.parquet"
# output_file_path = "mini_data/pt/accommodation_catering_hotel/chinese/high/rank_00000.parquet"

# input_file_path = "data/pt/accommodation_catering_hotel/english/high/rank_00726.parquet"
# output_file_path = "mini_data/pt/accommodation_catering_hotel/english/high/rank_00726.parquet"

# input_file_path = "data/sft/7M/train-00000-of-00075.parquet"
# output_file_path = "mini_data/sft/7M/train-00000-of-00075.parquet"

# input_file_path = "data/sft/Gen/train-00000-of-00015.parquet"
# output_file_path = "mini_data/sft/Gen/train-00000-of-00015.parquet"

# input_file_path = "data/dpo/train-00000-of-00001.parquet"
# output_file_path = "mini_data/dpo/train-00000-of-00001.parquet"

input_file_path = "data/dpo/test-00000-of-00001.parquet"
output_file_path = "mini_data/dpo/test-00000-of-00001.parquet"

# 读取Parquet文件
df = pd.read_parquet(input_file_path)

# 截取前100行
df_mini = df.head(100)

# 创建输出目录（如果不存在）
import os

os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# 将截取的数据保存到新的Parquet文件
df_mini.to_parquet(output_file_path)
