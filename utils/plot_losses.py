import os
import matplotlib.pyplot as plt


def extract_and_add_step(log_file):
    entries_with_step = []
    step_counter = 20

    with open(log_file, "r", encoding="utf-8") as file:
        for line in file:
            try:
                # 尝试解析每一行
                entry = eval(line)
                # 检查是否包含所需键值对
                if all(
                    key in entry
                    for key in ["loss", "grad_norm", "learning_rate", "epoch"]
                ):
                    # 添加或更新step字段
                    entry["step"] = step_counter
                    # 增加step计数器
                    step_counter += 20
                    # 将处理后的条目添加到列表中
                    entries_with_step.append(entry)
            except:
                # 如果解析失败，则跳过该行
                continue

    return entries_with_step


def load_log_histories(directory, filenames):
    """加载多个JSON文件的日志历史"""
    log_histories = []
    for filename in filenames:
        log_history = extract_and_add_step(os.path.join(directory, filename))
        log_histories.append(log_history)  # log_history[:11]为局部数据
    return log_histories


def plot_loss(save_directory, log_histories, mode):
    """绘制训练损失曲线并保存图像"""
    plt.switch_backend("agg")  # 使用非交互式后端
    key = "loss"  # 默认使用 "loss" 作为绘图的关键字

    # 设置颜色列表和标签列表以区分不同曲线
    colors = ["#ff7f0e", "#2ca02c", "#d62728", "#1f77b4"]
    labels = [f"{mode}1", f"{mode}2", f"{mode}3", f"{mode}"]

    plt.figure()
    for idx, log_history in enumerate(log_histories):
        steps, metrics = [], []

        # 提取损失数据
        for log_entry in log_history:
            steps.append(log_entry["step"])
            metrics.append(log_entry[key])

        # 绘制每条曲线
        plt.plot(steps, metrics, color=colors[idx], label=labels[idx])

    # 设置标题和坐标轴标签
    # plt.title(f"Training {mode} {key} Comparison (Part)")
    plt.title(f"Training {mode} {key} Comparison")
    plt.xlabel("Step")
    plt.ylabel(key.capitalize())
    plt.legend()

    # 保存图像
    # figure_path = os.path.join(save_directory, f"training_{key.replace('/', '_')}_{mode}_comparison_part.png")
    figure_path = os.path.join(save_directory, f"training_{key.replace('/', '_')}_{mode}_comparison.png")
    plt.savefig(figure_path, format="png", dpi=100)
    print(f"Figure saved at: {figure_path}")


# 目录和文件名列表
directory = "logs"
filenames = ["output_pt1.log", "output_pt2.log", "output_pt3.log", "output_pt.log"]
mode = "pt"
# filenames = ["output_dpo1.log", "output_dpo2.log", "output_dpo3.log", "output_dpo.log"]
# mode = "dpo"

# 加载日志历史
log_histories = load_log_histories(directory, filenames)

# 图片保存路径
output_path = "images"

# 绘制并保存损失曲线
plot_loss(output_path, log_histories, mode)
