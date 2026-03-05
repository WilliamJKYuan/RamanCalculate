# Code Verion 3.3.1
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 强制非交互式后端，防止画图时内存泄漏
import matplotlib.pyplot as plt
from scipy import stats
import os
from itertools import combinations, product
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
import time
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 设置英文字体
try:
    plt.rcParams['font.family'] = ['DejaVu Serif', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    plt.rcParams['font.sans-serif'] = ['DejaVu Serif', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

def get_target_indices(input_str, raman_shifts, mean_intensities):
    """
    解析用户输入，支持单点精确/就近匹配和范围匹配
    如果输入的是范围，则在该范围内寻找平均强度最高的峰值点
    """
    if not input_str.strip():
        return []
    
    indices = set()
    # 净化字符串
    input_str = input_str.replace('，', ',').replace('(', '').replace(')', '').replace('[', '').replace(']', '')
    parts = input_str.split(',')
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # 识别范围 (包含 '-' 且不是负数开头)
        if '-' in part and not part.startswith('-'):
            try:
                start_str, end_str = part.split('-')
                start, end = sorted([float(start_str.strip()), float(end_str.strip())])
                
                # 获取范围内的所有候选索引
                in_range_mask = (raman_shifts >= start) & (raman_shifts <= end)
                range_indices = np.where(in_range_mask)[0]
                
                if len(range_indices) > 0:
                    # 在该范围内的索引中找到平均强度最大的那个（即峰值点）
                    range_intensities = mean_intensities[range_indices]
                    peak_local_idx = np.nanargmax(range_intensities)
                    peak_global_idx = range_indices[peak_local_idx]
                    
                    indices.add(peak_global_idx)
                    print(f"提示：范围 {start}-{end} 匹配到峰值点，位于偏移量 {raman_shifts[peak_global_idx]:.3f}")
                else:
                    print(f"警告：范围 {start}-{end} 内未找到任何拉曼偏移量。")
            except ValueError:
                print(f"无法解析范围 '{part}'，跳过。")
        else:
            # 识别单点
            try:
                val = float(part)
                # 寻找最接近的实际拉曼偏移索引
                idx = (np.abs(raman_shifts - val)).argmin()
                indices.add(idx)
                actual_val = raman_shifts[idx]
                if abs(actual_val - val) > 2.0:  # 如果偏差过大给个提示
                    print(f"提示：单点 {val} 匹配到最接近的值为 {actual_val:.3f} (偏差较大)")
            except ValueError:
                print(f"无法解析单点 '{part}'，跳过。")
                
    return sorted(list(indices))

def get_all_subsets(indices, max_len=None):
    """
    获取索引列表的所有非空子集组合
    """
    subsets = []
    max_r = len(indices) if max_len is None else min(len(indices), max_len)
    for r in range(1, max_r + 1):
        subsets.extend(list(combinations(indices, r)))
    return subsets

def calculate_combination(args):
    """
    第一阶段：纯数学计算（不包含任何绘图逻辑）
    """
    indices_num, indices_den, raman_shifts, concentrations, intensities_matrix, use_log, itns_log = args
    
    # 将多个峰的强度按列相加
    intensity_num = np.sum(intensities_matrix[list(indices_num)], axis=0)
    intensity_den = np.sum(intensities_matrix[list(indices_den)], axis=0)
    
    # 构造组合标签名
    num_label = "+".join([f"{raman_shifts[i]:.2f}" for i in indices_num])
    den_label = "+".join([f"{raman_shifts[i]:.2f}" for i in indices_den])
    comb_label = f"({num_label}) / ({den_label})"
    
    # 数学计算
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ratio = intensity_num / intensity_den
        ratio = np.where(ratio > 0, ratio, np.nan)
        if itns_log in ['y', 'yes', '是']:
            intensity_diff = np.log10(ratio)
        else:
            intensity_diff = ratio

    valid_mask = np.isfinite(concentrations) & np.isfinite(intensity_diff)
    clean_conc = concentrations[valid_mask]
    clean_diff = intensity_diff[valid_mask]
    
    if len(clean_conc) < 3:
        return {
            '组合': comb_label,
            '斜率': np.nan, '截距': np.nan, 'R_squared': np.nan, 'P值': np.nan,
            'num_indices': indices_num, 'den_indices': indices_den, 'valid_pts': len(clean_conc)
        }
        
    slope, intercept, r_value, p_value, std_err = stats.linregress(clean_conc, clean_diff)
    
    return {
        '组合': comb_label,
        '斜率': slope,
        '截距': intercept,
        'R_squared': r_value**2, 
        'P值': p_value,
        'num_indices': indices_num, 
        'den_indices': indices_den,
        'valid_pts': len(clean_conc)
    }

def plot_combination(args):
    """
    第二阶段：仅针对达到 R_squared 阈值的组合进行绘图
    """
    row_dict, raman_shifts, concentrations, intensities_matrix, output_dir, use_log, itns_log = args
    
    indices_num = row_dict['num_indices']
    indices_den = row_dict['den_indices']
    comb_label = row_dict['组合']
    r_squared = row_dict['R_squared']
    slope = row_dict['斜率']
    intercept = row_dict['截距']
    p_value = row_dict['P值']
    valid_pts = row_dict['valid_pts']
    
    intensity_num = np.sum(intensities_matrix[list(indices_num)], axis=0)
    intensity_den = np.sum(intensities_matrix[list(indices_den)], axis=0)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ratio = intensity_num / intensity_den
        ratio = np.where(ratio > 0, ratio, np.nan)
        intensity_diff = np.log10(ratio) if itns_log in ['y', 'yes', '是'] else ratio

    valid_mask = np.isfinite(concentrations) & np.isfinite(intensity_diff)
    clean_conc = concentrations[valid_mask]
    clean_diff = intensity_diff[valid_mask]

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(clean_conc, clean_diff, alpha=0.7, s=50, 
              label='Experimental data', color='blue', edgecolors='black', linewidths=0.5)
    
    x_fit = np.linspace(clean_conc.min(), clean_conc.max(), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Linear fit (R_squared = {r_squared:.4f})')
    
    # 动态调整标题大小以适应超长文本
    title_fontsize = 10 if len(comb_label) > 60 else 14
    
    if use_log in ['y', 'yes', '是']:
        ax.set_xlabel('Log10 Concentration', fontsize=12)
        ax.set_ylabel(f'Log10 Intensity Ratio', fontsize=12)
        ax.set_title(f'Log10 Ratio: {comb_label}', fontsize=title_fontsize, fontweight='bold', wrap=True)
    else:
        ax.set_xlabel('Concentration', fontsize=12)
        ax.set_ylabel(f'Intensity Ratio', fontsize=12)
        ax.set_title(f'Ratio: {comb_label}', fontsize=title_fontsize, fontweight='bold', wrap=True)    
    
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    textstr = f'Slope: {slope:.4f}\nIntercept: {intercept:.4f}\nR_squared: {r_squared:.4f}\nP-value: {p_value:.4e}\nValid Pts: {valid_pts}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    safe_r_squared_category = f"r2_{int(r_squared * 10)}"
    category_dir = os.path.join(output_dir, 'scatter_plots', safe_r_squared_category)
    os.makedirs(category_dir, exist_ok=True)
    
    # 文件名防过长处理
    safe_name = comb_label.replace('/', 'div').replace('+', '_').replace('(', '').replace(')', '').replace(' ', '')
    if len(safe_name) > 100:
        safe_name = safe_name[:90] + "_etc"
    filename = f"{safe_name}_r2_{r_squared:.4f}.png"
    filepath = os.path.join(category_dir, filename)
    
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    fig.clf()
    plt.close(fig)
    plt.close('all')

def main():
    csv_file = input("输入文件路径：").strip(' "\'').replace('\\', '/')
    output_dir = 'raman_analysis_multithreaded'
    
    if not os.path.exists(csv_file):
        print(f"错误：文件 {csv_file} 不存在！")
        return
        
    print(f"\n正在读取文件...")
    df = pd.read_csv(csv_file, index_col=0)
    concentrations = df.columns.astype(float).values
    raman_shifts = df.index.astype(float).values
    intensities_matrix = df.values
    
    invalid_mask = intensities_matrix <= 0
    if np.sum(invalid_mask) > 0:
        intensities_matrix = np.where(intensities_matrix > 0, intensities_matrix, np.nan)
        print("已自动将非正数强度转为 NaN。")

    # ======= 新增：计算平均强度光谱用于寻峰 =======
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 按行计算所有浓度下强度的平均值，忽略 NaN
        mean_intensities = np.nanmean(intensities_matrix, axis=1)
    # ===============================================
    
    use_log = input("是否使用对数浓度？(y/n, 默认为y): ").strip().lower() or 'y'
    itns_log = input("是否使用对数强度比值？(y/n, 默认为y):").strip().lower() or 'y'

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cct_cacul = np.log10(concentrations) if use_log in ['y', 'yes', '是'] else concentrations

    # ======= 修改：传入 mean_intensities 参数 =======
    str_a = input("\n请输入A组(刃天青)的偏移量或范围 (例: 704.3, 968.9, 1100-1200): ").strip()
    indices_a = get_target_indices(str_a, raman_shifts, mean_intensities)
    print(f"A组成功匹配到 {len(indices_a)} 个真实偏移节点。")
    
    str_b = input("\n请输入B组(试卤灵)的偏移量或范围 (例: 571.6, 717.1, 1200-1300): ").strip()
    indices_b = get_target_indices(str_b, raman_shifts, mean_intensities)
    print(f"B组成功匹配到 {len(indices_b)} 个真实偏移节点。")
    # ===============================================

    if not indices_a or not indices_b:
        print("错误：A组或B组未匹配到任何数据，程序终止。")
        return

    max_len_input = input("\n单个组内最多允许几个峰相加？(直接回车表示不限制，推荐输入 3 或 4 防止组合爆炸): ").strip()
    max_len = int(max_len_input) if max_len_input.isdigit() else None

    subsets_a = get_all_subsets(indices_a, max_len)
    subsets_b = get_all_subsets(indices_b, max_len)
    
    combinations_list = list(product(subsets_a, subsets_b)) + list(product(subsets_b, subsets_a))
    total_combinations = len(combinations_list)
    
    print(f"\n排列组合构建完毕：A组提取 {len(subsets_a)} 种子集，B组提取 {len(subsets_b)} 种子集。")
    print(f"总计需要进行 {total_combinations} 次交叉比值拟合计算！")
    
    if total_combinations > 5000000:
        cont = input("警告：组合数量极大，可能需要较长内存和时间。是否继续？(y/n): ")
        if cont.lower() != 'y':
            return

    # ================= 阶段 1：纯计算 =================
    start_time = time.time()
    max_workers_input = input("设置进程数 (默认全CPU线程-2): ").strip()
            
    max_workers = multiprocessing.cpu_count() - 2
    if max_workers_input:
        try:
            max_workers = int(max_workers_input)
        except ValueError:
            print("输入无效，将自动使用默认核心数")
    
    
    args_list = [
        (num, den, raman_shifts, cct_cacul, intensities_matrix, use_log, itns_log)
        for num, den in combinations_list
    ]
    
    results = []
    print(f"\n[阶段 1/2] 开始核心多进程计算 (使用 {max_workers} 个进程)...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        chunk_size = max(1, total_combinations // (max_workers * 20))
        results_iterator = executor.map(calculate_combination, args_list, chunksize=chunk_size)
        
        with tqdm(total=total_combinations, desc="数学拟合", unit="项") as pbar:
            for res in results_iterator:
                results.append(res)
                pbar.update(1)
                
    elapsed_time = time.time() - start_time
    print(f"\n计算阶段完成！耗时: {elapsed_time:.2f} 秒")
    
    # 转换为DataFrame进行保存
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame(results)
    
    # 清洗并保存完整CSV (移除绘图专用的内部索引元组列，使CSV干净可读)
    csv_df = results_df.drop(columns=['num_indices', 'den_indices'])
    csv_df = csv_df.sort_values('R_squared', ascending=False, na_position='last')
    
    result_csv = os.path.join(output_dir, 'all_combinations_results.csv')
    csv_df.to_csv(result_csv, index=False, encoding='utf-8-sig')
    print(f"全量计算结果已导出至: {result_csv}")
    
    valid_r2 = csv_df['R_squared'].dropna()
    if not valid_r2.empty:
        print(f"\n数据概览: 最优 R_squared = {valid_r2.max():.6f}, 平均 = {valid_r2.mean():.6f}")
        
        # ================= 新增：绘制 R 方分布图 =================
        fig = plt.figure(figsize=(12, 6))
        
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.hist(valid_r2, bins=min(30, len(valid_r2)), edgecolor='black', alpha=0.7, color='skyblue')
        ax1.set_xlabel('R_squared Value', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('R_squared Value Distribution Histogram', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        textstr = f'Valid Fits: {len(valid_r2)}\nMean: {valid_r2.mean():.4f}\nMedian: {valid_r2.median():.4f}\nStd: {valid_r2.std():.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.98, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.boxplot(valid_r2, vert=True, patch_artist=True)
        ax2.set_ylabel('R_squared Value', fontsize=12)
        ax2.set_title('R_squared Value Distribution Boxplot', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xticklabels(['Selected permutations'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'r_squared_distribution.png'), dpi=150, bbox_inches='tight')
        fig.clf()
        plt.close(fig)
        print(f"R 方分布图已保存到: {os.path.join(output_dir, 'r_squared_distribution.png')}")
        # ========================================================
    else:
        print("未拟合出任何有效数据，跳过分布图绘制。")
        return

    # ================= 保存文本总结报告 =================
    summary_file = os.path.join(output_dir, 'analysis_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("处理报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"设定的A组(刃天青)输入: {str_a}\n")
        f.write(f"设定的B组(试卤灵)输入: {str_b}\n")
        f.write(f"单个组内最多允许相加峰数: {max_len_input if max_len_input else '不限制'}\n\n")
        f.write(f"分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总排列数: {total_combinations}\n")
        f.write(f"核心计算执行时间: {elapsed_time:.2f} 秒\n")
        
        if not valid_r2.empty:
            f.write("\n最佳20个组合:\n")
            f.write("-" * 30 + "\n")
            for idx, row in csv_df.head(20).iterrows():
                f.write(f"组合: {row['组合']}\n")
                f.write(f"  R_squared: {row['R_squared']:.6f}\n")
                f.write(f"  斜率: {row['斜率']:.6f}\n")
                f.write(f"  截距: {row['截距']:.6f}\n")
                f.write(f"  P值: {row['P值']:.6e}\n\n")
            
    print(f"txt 报告已保存到: {summary_file}")
    # ========================================================

    # ================= 阶段 2：选择性绘图 =================
    plot_input = input("\n[阶段 2/2] 请输入需要导出散点图的最低 R_squared 阈值 (例如 0.85，输入 'n' 直接退出): ").strip()
    if plot_input.lower() in ['n', 'no']:
        print("跳过绘图阶段，程序结束。")
        return
        
    try:
        r2_threshold = float(plot_input)
    except ValueError:
        print("阈值输入无效，默认只画 R_squared >= 0.8 的图。")
        r2_threshold = 0.8
        
    # 筛选达标的原始字典结果(保留用于传参的内部索引)
    plot_targets = [res for res in results if pd.notna(res['R_squared']) and res['R_squared'] >= r2_threshold]
    
    print(f"共有 {len(plot_targets)} 个组合的 R_squared 大于等于 {r2_threshold}。开始并行绘图...")
    
    if len(plot_targets) > 0:
        plot_args_list = [
            (target, raman_shifts, cct_cacul, intensities_matrix, output_dir, use_log, itns_log)
            for target in plot_targets
        ]
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            plot_iterator = executor.map(plot_combination, plot_args_list)
            with tqdm(total=len(plot_targets), desc="生成图片", unit="张") as pbar:
                for _ in plot_iterator:
                    pbar.update(1)
        
        print(f"\n绘图完成！图片已保存在 {output_dir}/scatter_plots 目录下。")
    else:
        print("没有达到该阈值的组合，跳过绘图。")

if __name__ == "__main__":
    main()
