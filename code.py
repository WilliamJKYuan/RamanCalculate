# Code Version 3.1.4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from itertools import permutations, product
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
import time
from tqdm import tqdm
import sys
import argparse

warnings.filterwarnings('ignore')

# 设置英文字体
try:
    plt.rcParams['font.family'] = ['DejaVu Serif', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    plt.rcParams['font.sans-serif'] = ['DejaVu Serif', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

def parse_ranges(input_str):
    """
    范围字符串转为元组列表
    输入: '1-10, 30-40' -> 输出: [(1.0, 10.0), (30.0, 40.0)]
    """
    if not input_str.strip():
        return []
    
    ranges = []
    # 替换中文逗号为英文逗号，便于分割
    input_str = input_str.replace('，', ',')
    parts = input_str.split(',')
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            start_str, end_str = part.split('-')
            start = float(start_str.strip())
            end = float(end_str.strip())
            # 自动纠正大小顺序，确保 start <= end
            if start > end:
                start, end = end, start
            ranges.append((start, end))
        except ValueError:
            print(f"无法解析范围 '{part}'，请确保格式如 '1-10'。已跳过该范围。")
            
    return ranges

def process_single_combination(args):
    """
    处理单个组合
    """
    i, j, raman_shifts, concentrations, intensities_matrix, output_dir, use_log = args, itns_log = args
    
    raman_i = raman_shifts[i]
    raman_j = raman_shifts[j]
    
    # 获取两个拉曼偏移的强度数组
    intensity_i = intensities_matrix[i]
    intensity_j = intensities_matrix[j]
    
    # 处理 NaN 与除以 0 的情况
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ratio = intensity_i / intensity_j
        ratio = np.where(ratio > 0, ratio, np.nan)
        
        if itns_log in ['y', 'yes', '是']:
            # 过滤掉非正数的比例以避免 log10 报错
            
            intensity_diff = np.log10(ratio)
        else:
            intensity_diff = ratio

    # 找到浓度和强度差值中都是有效数字 (Finite) 的索引
    valid_mask = np.isfinite(concentrations) & np.isfinite(intensity_diff)
    
    clean_concentrations = concentrations[valid_mask]
    clean_intensity_diff = intensity_diff[valid_mask]
    
    # 确保剩余的有效数据点足够进行线性拟合（至少需要3个点具备统计学意义）
    if len(clean_concentrations) < 3:
        return {
            '拉曼偏移分子': raman_i,
            '拉曼偏移分母': raman_j,
            '组合': f"{raman_i:.2f} / {raman_j:.2f}",
            '斜率': np.nan,
            '截距': np.nan,
            'R值': np.nan,
            'R_squared': np.nan, 
            'P值': np.nan,
            '标准误差': np.nan,
            '图片路径': f'有效数据不足(剩余{len(clean_concentrations)}个)，放弃拟合'
        }

    # 线性回归计算
    slope, intercept, r_value, p_value, std_err = stats.linregress(clean_concentrations, clean_intensity_diff)
    r_squared = r_value ** 2
    
    # 防止最终 R_squared 依然为 NaN 导致的强制转换整数报错
    if np.isnan(r_squared):
        safe_r_squared_category = "r2_nan"
    else:
        safe_r_squared_category = f"r2_{int(r_squared * 10)}"
    
    # 绘制散点图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制散点 (使用清洗后的有效数据)
    ax.scatter(clean_concentrations, clean_intensity_diff, alpha=0.7, s=50, 
              label='Experimental data', color='blue', edgecolors='black', linewidths=0.5)
    
    # 绘制拟合直线
    x_fit = np.linspace(clean_concentrations.min(), clean_concentrations.max(), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'r-', linewidth=2, 
            label=f'Linear fit (R_squared = {r_squared:.4f})')
    
    # 图表属性设置
    if use_log in ['y', 'yes', '是']:
        ax.set_xlabel('Log10 Concentration', fontsize=12)
        ax.set_ylabel(f'Log10 Intensity ratio (Shift {raman_i:.2f} / {raman_j:.2f})', fontsize=12)
        ax.set_title(f'Log10 Intensity Ratio Analysis: {raman_i:.2f} / {raman_j:.2f}', 
                fontsize=14, fontweight='bold')
    else:
        ax.set_xlabel('Concentration', fontsize=12)
        ax.set_ylabel(f'Intensity ratio (Shift {raman_i:.2f} / {raman_j:.2f})', fontsize=12)
        ax.set_title(f'Intensity Ratio Analysis: {raman_i:.2f} / {raman_j:.2f}', 
                fontsize=14, fontweight='bold')    
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加拟合参数文本框
    textstr = f'Slope: {slope:.4f}\nIntercept: {intercept:.4f}\nR_squared: {r_squared:.4f}\nP-value: {p_value:.4e}\nValid Points: {len(clean_concentrations)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # 创建子目录
    category_dir = os.path.join(output_dir, 'scatter_plots', safe_r_squared_category)
    os.makedirs(category_dir, exist_ok=True)
    
    # 更新文件名
    filename = f"{raman_i:.2f}_div_{raman_j:.2f}_r2_{r_squared:.4f}.png"
    filepath = os.path.join(category_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        '拉曼偏移分子': raman_i,
        '拉曼偏移分母': raman_j,
        '组合': f"{raman_i:.2f} / {raman_j:.2f}",
        '斜率': slope,
        '截距': intercept,
        'R值': r_value,
        'R_squared': r_squared, 
        'P值': p_value,
        '标准误差': std_err,
        '图片路径': filepath
    }

def analyze_raman_data_multithreaded(csv_file, ranges_a, ranges_b, output_dir='raman_analysis', 
                                     max_workers=None, use_processes=True, use_log="y", itns_log="y"):
    """
    分析拉曼光谱数据并支持多线程/多进程加速
    """
    start_time = time.time()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    scatter_base_dir = os.path.join(output_dir, 'scatter_plots')
    if not os.path.exists(scatter_base_dir):
        os.makedirs(scatter_base_dir)
    
    print(f"\n正在读取文件: {csv_file}")
    df = pd.read_csv(csv_file, index_col=0)
    
    # 获取浓度值
    concentrations = df.columns.astype(float).values
    
    # 如果输入的表头有 <= 0 的情况，提前规避 log10 的警告
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logcct = np.log10(concentrations)

    #log or not
    if use_log in ['y', 'yes', '是']:
        cct_cacul = logcct
        print("使用对数浓度进行计算")
    else:
        cct_cacul = concentrations
        print("不使用对数浓度计算")

    raman_shifts = df.index.astype(float).values
    intensities_matrix = df.values
    
    # 数据检查
    invalid_mask = intensities_matrix <= 0
    invalid_count = np.sum(invalid_mask)
    
    if invalid_count > 0:
        total_data_points = intensities_matrix.size
        invalid_ratio = (invalid_count / total_data_points) * 100
        print(f"\n数据检查：检查到 {invalid_count} 个负数或零拉曼强度值，占比 {invalid_ratio:.2f}%。")
        
        # 将所有 <= 0 的值替换为 np.nan，确保它们不参与后续数学运算
        intensities_matrix = np.where(intensities_matrix > 0, intensities_matrix, np.nan)
        print("已将非正数强度剔除（转为 NaN）。")
    
    print(f"\n数据形状: {df.shape}")
    print(f"原始浓度范围: {concentrations.min()} - {concentrations.max()}")
    print(f"拉曼偏移量总数量: {len(raman_shifts)}")
    
    # 基于输入范围的筛选逻辑
    def in_ranges(val, target_ranges):
        return any(start <= val <= end for start, end in target_ranges)
    
    indices_a = [i for i, shift_val in enumerate(raman_shifts) if in_ranges(shift_val, ranges_a)]
    indices_b = [i for i, shift_val in enumerate(raman_shifts) if in_ranges(shift_val, ranges_b)]
    
    print(f"\n根据设定的范围筛选结果：")
    print(f"刃天青范围 {ranges_a} 匹配到 {len(indices_a)} 个有效偏移值")
    print(f"试卤灵范围 {ranges_b} 匹配到 {len(indices_b)} 个有效偏移值")
    
    if len(indices_a) == 0 or len(indices_b) == 0:
        print("错误：刃天青或试卤灵没有匹配到任何数据，请检查设定的范围或原始CSV文件！")
        return pd.DataFrame()

    # 笛卡尔积组合
    combinations_list = list(product(indices_a, indices_b)) + list(product(indices_b, indices_a))
    total_combinations = len(combinations_list)
    
    print(f"\n总共需要分析 {total_combinations} 个组合")
    
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()-2
    
    print(f"使用 {max_workers} 个 {'进程' if use_processes else '线程'} 进行分析")
    
    args_list = []
    for i, j in combinations_list:
        args_list.append((i, j, raman_shifts, cct_cacul, 
                         intensities_matrix, scatter_base_dir, use_log, itns_log))
    
    executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    results = []
    
    # 并行计算并显示进度条
    with executor_class(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_combination, args) for args in args_list]
        with tqdm(total=total_combinations, desc="处理进度", unit="个") as pbar:
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"\n处理组合时出错: {str(e)}")
                finally:
                    pbar.update(1)
    
    print(f"\n散点图保存到: {scatter_base_dir}")
    
    # 处理结果并保存
    results_df = pd.DataFrame(results)
    if results_df.empty:
        return results_df
        
    backup_csv = os.path.join(output_dir, 'raw_results_backup.csv')
    results_df.to_csv(backup_csv, index=False, encoding='utf-8-sig')
    
    # na_position='last' 确保将废弃的 NaN 结果排在 CSV 文件的最后
    results_df = results_df.sort_values('R_squared', ascending=False, na_position='last')
    
    result_csv = os.path.join(output_dir, 'result.csv')
    results_df.to_csv(result_csv, index=False, encoding='utf-8-sig')
    print(f"排序结果已保存到: {result_csv}")
    
    elapsed_time = time.time() - start_time
    
    # 过滤出有效的 R_squared 用于统计和绘图
    valid_r2 = results_df['R_squared'].dropna()
    
    print(f"\n==== 分析完成 ====")
    print(f"执行耗时: {elapsed_time:.2f} 秒")
    if not valid_r2.empty:
        print(f"最优排列: {results_df.iloc[0]['组合']}, R_squared = {valid_r2.max():.6f}")
        print(f"最差排列: {results_df.iloc[len(valid_r2)-1]['组合']}, R_squared = {valid_r2.min():.6f}")
        print(f"平均R_squared: {valid_r2.mean():.6f}")
        
        # 绘制 R 方分布图 (仅使用有效数据)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax1 = plt.subplot(1, 2, 1)
        ax1.hist(valid_r2, bins=min(30, len(valid_r2)), edgecolor='black', alpha=0.7, color='skyblue')
        ax1.set_xlabel('R_squared Value', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('R_squared Value Distribution Histogram', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        textstr = f'Valid Fits: {len(valid_r2)}\nMean: {valid_r2.mean():.4f}\nMedian: {valid_r2.median():.4f}\nStd: {valid_r2.std():.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.98, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        ax2 = plt.subplot(1, 2, 2)
        ax2.boxplot(valid_r2, vert=True, patch_artist=True)
        ax2.set_ylabel('R_squared Value', fontsize=12)
        ax2.set_title('R_squared Value Distribution Boxplot', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xticklabels(['Selected permutations'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'r_squared_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        print("\n所有排列均未能生成有效的 R_squared，未生成分布图。")
    
    # 保存文本总结报告
    summary_file = os.path.join(output_dir, 'analysis_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("处理报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"设定的刃天青范围: {ranges_a}\n")
        f.write(f"设定的试卤灵范围: {ranges_b}\n\n")
        f.write(f"分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总排列数: {total_combinations}\n")
        f.write(f"执行时间: {elapsed_time:.2f} 秒\n")
        
        if not valid_r2.empty:
            f.write("\n最佳20个组合:\n")
            f.write("-" * 30 + "\n")
            for idx, row in results_df.head(20).iterrows():
                f.write(f"组合: {row['组合']}\n")
                f.write(f"  R_squared: {row['R_squared']:.6f}\n")
                f.write(f"  斜率: {row['斜率']:.6f}\n")
                f.write(f"  截距: {row['截距']:.6f}\n")
                f.write(f"  P值: {row['P值']:.6e}\n\n")
            
    print(f"txt 报告已保存到: {summary_file}")
    
    return results_df

def benchmark_parallel_methods(csv_file, ranges_a, ranges_b):
    """
    性能测试模式
    """
    print("Starting performance test...")
    configs = [
        ("Single-threaded", False, 1),
        ("Multithreading (4)", True, 4),
        ("Multiprocessing (4)", False, 4),
        ("Multithreading (all)", True, None),
        ("Multiprocessing (all)", False, None)
    ]
    results = {}
    
    for name, use_threads, workers in configs:
        print(f"\nTesting: {name}")
        start = time.time()
        output_dir = f'raman_analysis_benchmark_{name.replace(" ", "_")}'
        
        try:
            analyze_raman_data_multithreaded(
                csv_file, ranges_a, ranges_b, 
                output_dir=output_dir,
                max_workers=workers,
                use_processes=not use_threads
            )
            elapsed = time.time() - start
            results[name] = elapsed
            print(f"{name} completed in: {elapsed:.2f} seconds")
        except Exception as e:
            print(f"{name} failed: {str(e)}")
            
    if results:
        print("\nPerformance test results:")
        for name, time_val in results.items():
            print(f"{name}: {time_val:.2f} seconds")

def main():
    """
    主函数
    """
    csv_file = input("输入文件路径：").strip(' "\'').replace('\\', '/')
    output_dir = 'raman_analysis_multithreaded'
    
    if not os.path.exists(csv_file):
        print(f"错误：文件 {csv_file} 不存在！")
        print("请检查路径是否正确。")
        return
    
    use_log = input("是否使用对数浓度？(y/n, 默认为y): ").strip().lower() or 'y'
    itns_log = input("是否使用对数强度比值？(y/n, 默认为y):").strip().lower or 'y'

    ranges_a, ranges_b = [], []
    
    while not ranges_a:
        str_a = input("\n请输入刃天青的偏移范围aa-bb, cc-dd（英文逗号分隔）: ").strip()
        ranges_a = parse_ranges(str_a)
        if not ranges_a:
            print("输入无效，请重新输入正确的范围格式")
            
    while not ranges_b:
        str_b = input("请输入试卤灵的偏移范围aa-bb, cc-dd（英文逗号分隔）: ").strip()
        ranges_b = parse_ranges(str_b)
        if not ranges_b:
            print("输入无效，请重新输入正确的范围格式")

    print("\n请选择运行模式:")
    print("1. 普通并行分析")
    print("2. 性能测试 (比较不同并行计算方法的速度)")
    
    mode = input("请输入选项 (1 或 2，默认为1): ").strip() or '1'
    
    try:
        if mode == '2':
            benchmark_parallel_methods(csv_file, ranges_a, ranges_b)
        else:
            use_processes = input("\n使用多线程计算？(y/n，默认为y): ").strip().lower() != 'n'
            max_workers_input = input("设置进程数 (默认全CPU线程-2): ").strip()
            
            max_workers = None
            if max_workers_input:
                try:
                    max_workers = int(max_workers_input)
                except ValueError:
                    print("输入无效，将自动使用默认核心数")

            # 执行核心分析
            results = analyze_raman_data_multithreaded(
                csv_file, 
                ranges_a=ranges_a, 
                ranges_b=ranges_b,
                output_dir=output_dir,
                max_workers=max_workers,
                use_processes=use_processes,
                use_log=use_log,
                itns_log=itns_log
            )
            
            if not results.empty:
                print("\n=== 最佳20个组合 ===")
                # 过滤并展示有效的顶部结果
                valid_top_results = results.dropna(subset=['R_squared']).head(20)
                if not valid_top_results.empty:
                    print(valid_top_results[['组合', '斜率', '截距', 'R_squared', 'P值']].to_string(index=False))
                else:
                    print("没有成功拟合出有效的结果。")
                print(f"\n所有结果已保存。分析目录位于: {output_dir}")
            
    except Exception as e:
        print(f"\n 处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
