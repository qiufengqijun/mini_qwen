# # 下载模型 必要的文件已在项目中，可忽略
# cd models
# git lfs install
# git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
# cd ..

# 下载预训练数据集
modelscope download --dataset 'BAAI/IndustryCorpus2' --include 'film_entertainment/*/high*' --local_dir 'data/pt' # 数据量较大，英文文件选择前3个文件
find 'data/pt/film_entertainment/english/high' -maxdepth 1 -type f | sort | tail -n +4 | xargs rm -f

modelscope download --dataset 'BAAI/IndustryCorpus2' --include 'computer_programming_code/*/high*' --local_dir 'data/pt'
modelscope download --dataset 'BAAI/IndustryCorpus2' --include 'computer_communication/*/high*' --local_dir 'data/pt' # 数据量较大，英文文件选择前3个文件
find 'data/pt/computer_communication/english/high' -maxdepth 1 -type f | sort | tail -n +4 | xargs rm -f

modelscope download --dataset 'BAAI/IndustryCorpus2' --include 'tourism_geography/*/high*' --local_dir 'data/pt'
modelscope download --dataset 'BAAI/IndustryCorpus2' --include 'artificial_intelligence_machine_learning/*/high*' --local_dir 'data/pt'
modelscope download --dataset 'BAAI/IndustryCorpus2' --include 'news_media/*/high*' --local_dir 'data/pt'
modelscope download --dataset 'BAAI/IndustryCorpus2' --include 'literature_emotion/*/high*' --local_dir 'data/pt' # 数据量较大，英文文件选择前3个文件
find 'data/pt/literature_emotion/english/high' -maxdepth 1 -type f | sort | tail -n +4 | xargs rm -f

modelscope download --dataset 'BAAI/IndustryCorpus2' --include 'accommodation_catering_hotel/*/high*' --local_dir 'data/pt'
modelscope download --dataset 'BAAI/IndustryCorpus2' --include 'current_affairs_government_administration/*/high*' --local_dir 'data/pt' # 数据量较大，英文文件选择前3个文件
find 'data/pt/current_affairs_government_administration/english/high' -maxdepth 1 -type f | sort | tail -n +4 | xargs rm -f

modelscope download --dataset 'BAAI/IndustryCorpus2' --include 'mathematics_statistics/*/high*' --local_dir 'data/pt' # 数据量较大，英文文件选择前3个文件
find 'data/pt/mathematics_statistics/english/high' -maxdepth 1 -type f | sort | tail -n +4 | xargs rm -f

# 下载微调训练数据集
modelscope download --dataset 'BAAI/Infinity-Instruct' --local_dir 'data/sft' # 选择7M和Gen进行微调，因为这两个数据集更新时间最近，且数据量大

# 下载偏好数据集
modelscope download --dataset 'BAAI/Infinity-Preference' --local_dir 'data/dpo'
