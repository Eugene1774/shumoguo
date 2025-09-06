import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Set plot aesthetics
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei font for Chinese characters
plt.rcParams['axes.unicode_minus'] = False  # Ensure minus sign is displayed correctly
plt.style.use('seaborn-v0_8-whitegrid')

# Load the dataset
try:
    df = pd.read_csv('cleaned_data最终(1).csv')
except Exception as e:
    print(f"Error loading file: {e}")
    # Create a dummy dataframe for demonstration if file loading fails
    data = {'检测孕周_数值': np.random.randint(10, 25, 500),
            '13号染色体的Z值': np.random.randn(500) * 2,
            '18号染色体的Z值': np.random.randn(500) * 2,
            '21号染色体的Z值': np.random.randn(500) * 2,
            '染色体的非整倍体': [np.nan] * 480 + ['T21']*20,
            '胎儿是否健康': ['是'] * 480 + ['否']*20}
    df = pd.DataFrame(data)
    print("--- Using dummy data for demonstration ---")


# --- Step 1: Feature Engineering (Define NIPT result and Ground Truth) ---

# Define NIPT_Positive based on Z-scores or non-empty '染色体的非整倍体'
df['NIPT_Positive'] = (
    (df['13号染色体的Z值'].abs() > 3) |
    (df['18号染色体的Z值'].abs() > 3) |
    (df['21号染色体的Z值'].abs() > 3) |
    (df['染色体的非整倍体'].notna() & (df['染色体的非整倍体'].astype(str).str.strip() != ''))
)

# Define Is_Unhealthy based on the ground truth '胎儿是否健康'
# Assuming '否' means unhealthy. Adjust if your data uses different labels.
df['Is_Unhealthy'] = (df['胎儿是否健康'] == '否')

# Create a 'Category' column for confusion matrix components
def determine_category(row):
    if row['NIPT_Positive'] and row['Is_Unhealthy']:
        return 'TP'  # True Positive
    elif row['NIPT_Positive'] and not row['Is_Unhealthy']:
        return 'FP'  # False Positive
    elif not row['NIPT_Positive'] and row['Is_Unhealthy']:
        return 'FN'  # False Negative
    else:
        return 'TN'  # True Negative

df['Category'] = df.apply(determine_category, axis=1)

# Group data by gestational week to calculate rates
weekly_stats = df.groupby('检测孕周_数值')['Category'].value_counts().unstack(fill_value=0)

# Ensure all four columns (TP, FP, FN, TN) exist
for col in ['TP', 'FP', 'FN', 'TN']:
    if col not in weekly_stats.columns:
        weekly_stats[col] = 0

# --- Step 2: Calculate False Negative Rate (FNR) and False Positive Rate (FPR) ---

# FNR = FN / (FN + TP)  (The rate of missing actual unhealthy cases)
# Add a small epsilon to the denominator to avoid division by zero
epsilon = 1e-6
weekly_stats['FNR'] = weekly_stats['FN'] / (weekly_stats['FN'] + weekly_stats['TP'] + epsilon)

# FPR = FP / (FP + TN) (The rate of incorrectly flagging healthy cases)
weekly_stats['FPR'] = weekly_stats['FP'] / (weekly_stats['FP'] + weekly_stats['TN'] + epsilon)

# --- Step 3: Visualize the Risk Curves ---
fig, ax1 = plt.subplots(figsize=(14, 8))

# Plot FNR (漏诊率) - This is the most critical risk
ax1.plot(weekly_stats.index, weekly_stats['FNR'], 'o-', color='crimson', label='假阴性率 (FNR) - 漏诊风险', linewidth=2.5, markersize=8)
ax1.set_xlabel('检测孕周 (周)', fontsize=14)
ax1.set_ylabel('假阴性率 (FNR)', color='crimson', fontsize=14)
ax1.tick_params(axis='y', labelcolor='crimson', labelsize=12)
ax1.set_ylim(bottom=0)

# Create a second y-axis for FPR
ax2 = ax1.twinx()
ax2.plot(weekly_stats.index, weekly_stats['FPR'], 's--', color='darkorange', label='假阳性率 (FPR) - 误报风险', linewidth=2)
ax2.set_ylabel('假阳性率 (FPR)', color='darkorange', fontsize=14)
ax2.tick_params(axis='y', labelcolor='darkorange', labelsize=12)
ax2.set_ylim(bottom=0)


plt.title('NIPT误诊风险随检测孕周的变化趋势', fontsize=18, weight='bold')
fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('nipt_risk_curves.png', dpi=300)
plt.show()

# --- Step 4: Propose a new Objective Function ---
print("\n### 数据驱动的风险校准建议 ###")
print("1. 风险可视化: 请查看生成的 'nipt_risk_curves.png' 图。")
print("   - FNR (漏诊风险) 曲线是您决策的核心。我们期望在FNR进入一个稳定且足够低的平台期后，再进行检测。")
print("   - FPR (误报风险) 是次要考虑因素。")
print("\n2. 构建新的目标函数:")
print("   一个更合理的目标函数形式为: Total_Risk(t) = w_fn * FNR(t) + w_fp * FPR(t) + w_time * Risk_Time(t)")
print("   其中 't' 是检测孕周。FNR(t) 和 FPR(t) 可以通过对图中散点进行函数拟合得到。")

# --- Step 5: Calibrate Coefficients ---
# Calculate the scale of FNR and FPR
avg_fnr = weekly_stats[weekly_stats['FNR'] > 0]['FNR'].mean()
avg_fpr = weekly_stats[weekly_stats['FPR'] > 0]['FPR'].mean()

print("\n3. 风险系数校准建议:")
print(f"   - 数据中平均FNR约为: {avg_fnr:.4f}, 平均FPR约为: {avg_fpr:.4f}")
print("   - 临床上，漏诊(FN)的后果远重于误报(FP)。因此，权重 w_fn 应远大于 w_fp。")
print("     常见做法是设定一个重要性比率，例如 w_fn/w_fp = 10。我们可以暂定 w_fn = 10, w_fp = 1。")
print("   - 接下来，需要让 w_time * Risk_Time(t) 与误诊风险的尺度相当。")
print(f"     例如，当FNR为{avg_fnr:.4f}时，误诊风险项约为 10 * {avg_fnr:.4f} = {10*avg_fnr:.4f}。")
print("     您的时间风险系数是 1, 2, 4。如果直接使用，它们相对于误诊风险过大。")
print("     建议进行缩放，让最大时间风险(4)与一个不可接受的FNR风险大致相当。")
print("     例如，假设FNR > 0.1是绝对不可接受的，其风险值为 10 * 0.1 = 1。")
print("     那么，我们可以让晚期发现(28周后)的风险值也为1。这样，Risk_Time(t) 系数可以缩放为: [0.25, 0.5, 1.0]")
print("\n   - **最终建议的目标函数:**")
print("     Total_Risk(t) = 10 * FNR(t) + 1 * FPR(t) + Risk_Time_Scaled(t)")
print("     其中 Risk_Time_Scaled(t) 在12周前为0.25, 13-27周为0.5, 28周后为1.0。")
print("\n4. 后续步骤:")
print("   - 使用您的 'Y染色体浓度' 数据，为不同BMI分组，建立 P(检测成功 | t, BMI) 的模型。")
print("   - 您的最终优化目标函数是最小化期望总风险 E[Total_Risk]：")
print("     E[Total_Risk] = P(成功|t,BMI)*Total_Risk(t) + (1-P(成功|t,BMI))*Risk_失败(t)")
print("   - 使用 '等效风险法'，将 Risk_失败(t) 定义为 Risk_Time_Scaled(t + 延迟周数)，这样整个目标函数的单位就完全统一了。")