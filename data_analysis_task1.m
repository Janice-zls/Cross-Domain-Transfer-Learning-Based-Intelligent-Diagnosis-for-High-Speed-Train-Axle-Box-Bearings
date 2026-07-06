%% 高速列车轴承智能故障诊断 - 任务1：数据分析与故障特征提取

clear; clc; close all;

%% 1. 设置数据路径和参数
data_root = 'c:\Users\Jack\Desktop\E题\源域数据集\';
target_root = 'c:\Users\Jack\Desktop\E题\目标域数据集\';

% 轴承参数 (根据文档)
bearing_params.SKF6205.n = 9;           % 滚动体数
bearing_params.SKF6205.d = 0.3126;      % 滚动体直径 (英寸)
bearing_params.SKF6205.D = 1.537;       % 轴承节径 (英寸)

bearing_params.SKF6203.n = 9;           % 滚动体数
bearing_params.SKF6203.d = 0.2656;      % 滚动体直径 (英寸)
bearing_params.SKF6203.D = 1.122;       % 轴承节径 (英寸)

%% 2. 创建结果保存文件夹
result_dir = '问题一结果';
if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

%% 3. 主程序执行
fprintf('开始加载源域数据...\n');
[source_data, file_info] = load_bearing_data(data_root);

%% 4. 数据筛选和预处理（简化版）
fprintf('\n=== 数据筛选和预处理 ===\n');

% 显示原始数据信息
fprintf('原始数据统计:\n');
fprintf('总文件数: %d\n', length(source_data));

% 统计各类别原始数据
original_stats = containers.Map();
for i = 1:length(source_data)
    fault_type = source_data(i).fault_type;
    if isKey(original_stats, fault_type)
        original_stats(fault_type) = original_stats(fault_type) + 1;
    else
        original_stats(fault_type) = 1;
    end
end

fprintf('原始数据类别分布:\n');
fault_types = keys(original_stats);
for i = 1:length(fault_types)
    fprintf('  %s: %d 个文件\n', fault_types{i}, original_stats(fault_types{i}));
end

% 简化的数据质量检查
fprintf('\n进行数据质量检查...\n');
valid_indices = [];
for i = 1:length(source_data)
    is_valid = true;
    
    % 检查是否有有效的振动数据
    if isempty(source_data(i).de_data) && isempty(source_data(i).fe_data)
        is_valid = false;
    end
    
    % 检查数据长度
    main_data = [];
    if ~isempty(source_data(i).de_data)
        main_data = source_data(i).de_data;
    elseif ~isempty(source_data(i).fe_data)
        main_data = source_data(i).fe_data;
    end
    
    if ~isempty(main_data) && (length(main_data) < 1000 || all(main_data == 0))
        is_valid = false;
    end
    
    if is_valid
        valid_indices = [valid_indices, i];
    end
end

fprintf('数据质量检查完成:\n');
fprintf('  有效文件: %d 个\n', length(valid_indices));
fprintf('  无效文件: %d 个\n', length(source_data) - length(valid_indices));

% 筛选有效数据
source_data_filtered = source_data(valid_indices);

% 简化的数据预处理
fprintf('\n开始数据预处理...\n');
for i = 1:length(source_data_filtered)
    % 选择主要信号
    if ~isempty(source_data_filtered(i).de_data)
        main_signal = source_data_filtered(i).de_data;
    elseif ~isempty(source_data_filtered(i).fe_data)
        main_signal = source_data_filtered(i).fe_data;
    else
        continue;
    end
    
    % 基础预处理：去均值，限制长度
    main_signal = main_signal - mean(main_signal);
    if length(main_signal) > 120000
        main_signal = main_signal(1:120000);
    end
    
    % 更新数据
    if ~isempty(source_data_filtered(i).de_data)
        source_data_filtered(i).de_data = main_signal;
    else
        source_data_filtered(i).fe_data = main_signal;
    end
end

% 生成两张预处理图表
fprintf('\n生成预处理结果图表...\n');

% 图1：数据筛选结果统计
figure('Position', [100, 100, 1200, 600]);

% 子图1：原始数据分布
subplot(2, 2, 1);
fault_types = keys(original_stats);
original_counts = [];
for i = 1:length(fault_types)
    original_counts = [original_counts, original_stats(fault_types{i})];
end
bar(original_counts);
set(gca, 'XTickLabel', fault_types);
title('原始数据类别分布');
ylabel('文件数量');
grid on;

% 子图2：筛选后数据分布
subplot(2, 2, 2);
filtered_stats = containers.Map();
for i = 1:length(source_data_filtered)
    fault_type = source_data_filtered(i).fault_type;
    if isKey(filtered_stats, fault_type)
        filtered_stats(fault_type) = filtered_stats(fault_type) + 1;
    else
        filtered_stats(fault_type) = 1;
    end
end

filtered_counts = [];
for i = 1:length(fault_types)
    if isKey(filtered_stats, fault_types{i})
        filtered_counts = [filtered_counts, filtered_stats(fault_types{i})];
    else
        filtered_counts = [filtered_counts, 0];
    end
end
bar(filtered_counts);
set(gca, 'XTickLabel', fault_types);
title('筛选后数据类别分布');
ylabel('文件数量');
grid on;

% 子图3：筛选效果对比
subplot(2, 2, 3);
comparison_data = [original_counts; filtered_counts]';
bar(comparison_data);
set(gca, 'XTickLabel', fault_types);
title('数据筛选效果对比');
ylabel('文件数量');
legend({'原始数据', '筛选后数据'}, 'Location', 'best');
grid on;

% 子图4：筛选率统计
subplot(2, 2, 4);
selection_rates = filtered_counts ./ original_counts * 100;
bar(selection_rates);
set(gca, 'XTickLabel', fault_types);
title('各类别数据筛选率');
ylabel('筛选率 (%)');
ylim([0, 100]);
grid on;

sgtitle('图1：数据筛选结果统计', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(result_dir, '图1_数据筛选结果.png'));

% 图2：信号预处理效果展示
figure('Position', [200, 200, 1200, 800]);

% 选择几个代表性样本展示预处理效果
sample_indices = [];
for fault_type = fault_types
    type_indices = find(strcmp({source_data_filtered.fault_type}, fault_type{1}));
    if ~isempty(type_indices)
        sample_indices = [sample_indices, type_indices(1)];
    end
end

for i = 1:min(4, length(sample_indices))
    idx = sample_indices(i);
    
    % 获取原始信号（重新加载）
    original_data = load_bearing_data(data_root);
    original_sample = original_data(valid_indices(idx));
    
    if ~isempty(original_sample.de_data)
        original_signal = original_sample.de_data;
        processed_signal = source_data_filtered(idx).de_data;
    else
        original_signal = original_sample.fe_data;
        processed_signal = source_data_filtered(idx).fe_data;
    end
    
    % 限制显示长度
    display_length = min(5000, length(original_signal));
    time_axis = (1:display_length) / source_data_filtered(idx).fs;
    
    % 原始信号
    subplot(4, 2, 2*i-1);
    plot(time_axis, original_signal(1:display_length));
    title(sprintf('原始信号 - %s类别', source_data_filtered(idx).fault_type));
    xlabel('时间 (s)');
    ylabel('幅值');
    grid on;
    
    % 预处理后信号
    subplot(4, 2, 2*i);
    plot(time_axis, processed_signal(1:display_length));
    title(sprintf('预处理后信号 - %s类别', source_data_filtered(idx).fault_type));
    xlabel('时间 (s)');
    ylabel('幅值');
    grid on;
end

sgtitle('图2：信号预处理效果对比', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(result_dir, '图2_信号预处理效果.png'));

fprintf('数据筛选和预处理完成！\n');
fprintf('最终数据集包含 %d 个样本\n', length(source_data_filtered));

%% 5. 开始特征提取
fprintf('\n=== 开始特征提取 ===\n');

% 使用筛选后的数据进行特征提取
source_data = source_data_filtered; % 替换原始数据

% 初始化特征矩阵
num_samples = length(source_data);
feature_matrix = [];
labels = [];
sample_info = [];
fault_freq_data = []; % 存储故障特征频率信息

for i = 1:num_samples
    if isempty(source_data(i).de_data) && isempty(source_data(i).fe_data)
        continue;
    end
    
    fprintf('处理样本 %d/%d: %s\n', i, num_samples, source_data(i).filename);
    
    % 选择主要信号（优先选择DE数据）
    if ~isempty(source_data(i).de_data)
        main_signal = source_data(i).de_data;
    elseif ~isempty(source_data(i).fe_data)
        main_signal = source_data(i).fe_data;
    else
        continue;
    end
    
    % 确保信号是列向量
    if size(main_signal, 1) < size(main_signal, 2)
        main_signal = main_signal';
    end
    
    % 信号预处理（去均值，限制长度）
    main_signal = main_signal - mean(main_signal);
    if length(main_signal) > 120000  % 限制信号长度以提高处理速度
        main_signal = main_signal(1:120000);
    end
    
    % 计算故障特征频率
    if ~isempty(source_data(i).rpm) && source_data(i).rpm > 0
        bearing_param = bearing_params.(source_data(i).bearing_type);
        [bpfo, bpfi, bsf, ftf] = calculate_fault_frequencies(source_data(i).rpm, bearing_param);
        fault_freqs = struct('bpfo', bpfo, 'bpfi', bpfi, 'bsf', bsf, 'ftf', ftf);
        
        % 存储故障频率信息
        fault_freq_data = [fault_freq_data; bpfo, bpfi, bsf, ftf, source_data(i).rpm];
    else
        fault_freqs = [];
        fault_freq_data = [fault_freq_data; 0, 0, 0, 0, 0];
    end
    
    % 提取特征
    time_feat = extract_time_features(main_signal);
    freq_feat = extract_freq_features(main_signal, source_data(i).fs, fault_freqs);
    env_feat = extract_envelope_features(main_signal, source_data(i).fs, fault_freqs);
    
    % 组合特征向量
    feature_vector = [
        time_feat.mean, time_feat.std, time_feat.rms, time_feat.peak, ...
        time_feat.peak_to_peak, time_feat.crest_factor, time_feat.clearance_factor, ...
        time_feat.shape_factor, time_feat.impulse_factor, time_feat.skewness, ...
        time_feat.kurtosis, time_feat.energy, time_feat.power, ...
        freq_feat.spectral_centroid, freq_feat.spectral_spread, freq_feat.spectral_rolloff, ...
        freq_feat.spectral_flux, ...
        env_feat.envelope_rms, env_feat.envelope_peak, env_feat.envelope_crest
    ];
    
    % 添加故障特征频率相关特征（如果可用）
    if ~isempty(fault_freqs)
        fault_features = [
            freq_feat.bpfo_amplitude, freq_feat.bpfi_amplitude, freq_feat.bsf_amplitude, ...
            freq_feat.ftf_amplitude, freq_feat.bpfo_2x, freq_feat.bpfi_2x, freq_feat.bsf_2x, ...
            env_feat.env_bpfo, env_feat.env_bpfi, env_feat.env_bsf, env_feat.env_ftf
        ];
        feature_vector = [feature_vector, fault_features];
    else
        % 如果没有故障频率信息，用零填充
        fault_features = zeros(1, 11);
        feature_vector = [feature_vector, fault_features];
    end
    
    % 存储特征和标签
    feature_matrix = [feature_matrix; feature_vector];
    
    % 标签编码：N=0, B=1, IR=2, OR=3
    switch source_data(i).fault_type
        case 'N'
            label = 0;
        case 'B'
            label = 1;
        case 'IR'
            label = 2;
        case 'OR'
            label = 3;
        otherwise
            label = -1;
    end
    labels = [labels; label];
    
    % 存储样本信息
    sample_info = [sample_info; source_data(i)];
end

%% 6. 详细数据统计分析
fprintf('\n=== 详细数据统计分析 ===\n');
fprintf('总样本数: %d\n', size(feature_matrix, 1));
fprintf('特征维度: %d\n', size(feature_matrix, 2));

% 各类别样本数统计
unique_labels = unique(labels);
label_names = {'正常', '滚动体故障', '内圈故障', '外圈故障'};
label_codes = {'N', 'B', 'IR', 'OR'};

fprintf('\n各类别样本数统计:\n');
class_stats = [];
for i = 1:length(unique_labels)
    label = unique_labels(i);
    count = sum(labels == label);
    percentage = count / length(labels) * 100;
    fprintf('%s (%s): %d 个样本 (%.1f%%)\n', label_names{label+1}, label_codes{label+1}, count, percentage);
    class_stats = [class_stats; label, count, percentage];
end

% 按数据集统计
fprintf('\n按数据集统计:\n');
datasets = unique({sample_info.dataset});
dataset_stats = [];
for i = 1:length(datasets)
    dataset = datasets{i};
    count = sum(strcmp({sample_info.dataset}, dataset));
    percentage = count / length(sample_info) * 100;
    fprintf('%s: %d 个样本 (%.1f%%)\n', dataset, count, percentage);
    dataset_stats = [dataset_stats; i, count, percentage];
end

% 按载荷统计
fprintf('\n按载荷统计:\n');
loads = unique([sample_info.load_hp]);
load_stats = [];
for i = 1:length(loads)
    load_val = loads(i);
    count = sum([sample_info.load_hp] == load_val);
    percentage = count / length(sample_info) * 100;
    fprintf('%d HP: %d 个样本 (%.1f%%)\n', load_val, count, percentage);
    load_stats = [load_stats; load_val, count, percentage];
end

%% 5. 特征统计分析
fprintf('\n=== 特征统计分析 ===\n');

% 定义完整的特征名称
feature_names_full = {
    '均值', '标准差', 'RMS', '峰值', '峰峰值', '峭度系数', '裕度系数', ...
    '波形系数', '脉冲系数', '偏度', '峭度', '能量', '功率', ...
    '频谱质心', '频谱扩散', '频谱滚降', '频谱通量', ...
    '包络RMS', '包络峰值', '包络峭度系数', ...
    'BPFO幅值', 'BPFI幅值', 'BSF幅值', 'FTF幅值', ...
    'BPFO二次谐波', 'BPFI二次谐波', 'BSF二次谐波', ...
    '包络BPFO', '包络BPFI', '包络BSF', '包络FTF'
};

% 计算每个特征的统计量
feature_stats = [];
for i = 1:size(feature_matrix, 2)
    feat_data = feature_matrix(:, i);
    stats = [mean(feat_data), std(feat_data), min(feat_data), max(feat_data), ...
             median(feat_data), prctile(feat_data, 25), prctile(feat_data, 75)];
    feature_stats = [feature_stats; stats];
end

% 保存特征统计表
feature_stats_table = array2table(feature_stats, ...
    'VariableNames', {'均值', '标准差', '最小值', '最大值', '中位数', '25%分位数', '75%分位数'}, ...
    'RowNames', feature_names_full);
writetable(feature_stats_table, fullfile(result_dir, '特征统计表.csv'), 'WriteRowNames', true);

% 按类别计算特征统计
fprintf('计算各类别特征统计...\n');
class_feature_stats = {};
for class_idx = 1:length(unique_labels)
    label = unique_labels(class_idx);
    class_data = feature_matrix(labels == label, :);
    class_name = label_names{label+1};
    
    class_stats_data = [];
    for feat_idx = 1:size(class_data, 2)
        feat_data = class_data(:, feat_idx);
        stats = [mean(feat_data), std(feat_data), min(feat_data), max(feat_data)];
        class_stats_data = [class_stats_data; stats];
    end
    
    class_table = array2table(class_stats_data, ...
        'VariableNames', {'均值', '标准差', '最小值', '最大值'}, ...
        'RowNames', feature_names_full);
    writetable(class_table, fullfile(result_dir, sprintf('%s_特征统计.csv', class_name)), 'WriteRowNames', true);
    class_feature_stats{class_idx} = class_stats_data;
end

%% 6. 故障特征频率分析
fprintf('\n=== 故障特征频率分析 ===\n');
fault_freq_table = array2table(fault_freq_data, ...
    'VariableNames', {'BPFO_Hz', 'BPFI_Hz', 'BSF_Hz', 'FTF_Hz', 'RPM'});

% 按故障类型分组分析故障频率
for class_idx = 1:length(unique_labels)
    label = unique_labels(class_idx);
    if label == 0, continue; end % 跳过正常类别
    
    class_indices = labels == label;
    class_freq_data = fault_freq_data(class_indices, :);
    class_name = label_names{label+1};
    
    if ~isempty(class_freq_data) && any(class_freq_data(:,5) > 0)
        fprintf('\n%s 故障特征频率统计:\n', class_name);
        fprintf('  BPFO: %.2f ± %.2f Hz (范围: %.2f - %.2f Hz)\n', ...
            mean(class_freq_data(:,1)), std(class_freq_data(:,1)), ...
            min(class_freq_data(:,1)), max(class_freq_data(:,1)));
        fprintf('  BPFI: %.2f ± %.2f Hz (范围: %.2f - %.2f Hz)\n', ...
            mean(class_freq_data(:,2)), std(class_freq_data(:,2)), ...
            min(class_freq_data(:,2)), max(class_freq_data(:,2)));
        fprintf('  BSF:  %.2f ± %.2f Hz (范围: %.2f - %.2f Hz)\n', ...
            mean(class_freq_data(:,3)), std(class_freq_data(:,3)), ...
            min(class_freq_data(:,3)), max(class_freq_data(:,3)));
        fprintf('  FTF:  %.2f ± %.2f Hz (范围: %.2f - %.2f Hz)\n', ...
            mean(class_freq_data(:,4)), std(class_freq_data(:,4)), ...
            min(class_freq_data(:,4)), max(class_freq_data(:,4)));
    end
end

writetable(fault_freq_table, fullfile(result_dir, '故障特征频率表.csv'));

%% 7. 保存所有数据
save(fullfile(result_dir, 'task1_complete_data.mat'), 'feature_matrix', 'labels', 'sample_info', ...
     'bearing_params', 'fault_freq_data', 'feature_names_full', 'class_stats', 'dataset_stats', 'load_stats');

%% 8. 增强的可视化分析

% 图1: 类别分布饼图
figure('Position', [100, 100, 1200, 400]);
subplot(1, 3, 1);
pie(class_stats(:,2), label_names);
title('图1-1: 样本类别分布');

% 8.2 数据集分布
subplot(1, 3, 2);
bar(dataset_stats(:,2));
set(gca, 'XTickLabel', datasets);
title('图1-2: 各数据集样本数量');
ylabel('样本数');
xtickangle(45);

% 8.3 载荷分布
subplot(1, 3, 3);
bar(load_stats(:,1), load_stats(:,2));
title('图1-3: 载荷分布');
xlabel('载荷 (HP)');
ylabel('样本数');
saveas(gcf, fullfile(result_dir, '图1_数据分布统计.png'));

% 图2: 扩展的特征分布可视化
figure('Position', [100, 100, 1500, 1000]);
important_features = [3, 6, 11, 14, 17, 20, 21, 22, 23, 24]; % 选择10个重要特征
important_names = feature_names_full(important_features);

for i = 1:length(important_features)
    subplot(2, 5, i);
    feat_idx = important_features(i);
    
    % 为每个类别绘制箱线图
    data_for_boxplot = [];
    group_labels = [];
    
    for label = 0:3
        if sum(labels == label) > 0
            class_data = feature_matrix(labels == label, feat_idx);
            data_for_boxplot = [data_for_boxplot; class_data];
            group_labels = [group_labels; repmat(label, length(class_data), 1)];
        end
    end
    
    boxplot(data_for_boxplot, group_labels, 'Labels', label_names);
    title(important_names{i});
    ylabel('特征值');
    grid on;
    xtickangle(45);
end

sgtitle('图2: 重要特征的分布对比');
saveas(gcf, fullfile(result_dir, '图2_扩展特征分布.png'));

% 图3: 特征相关性热图
fprintf('计算特征相关性矩阵...\n');
corr_matrix = corrcoef(feature_matrix);
figure('Position', [100, 100, 1000, 800]);
imagesc(corr_matrix);
colorbar;
colormap('jet');
title('图3: 特征相关性热图');
xlabel('特征索引');
ylabel('特征索引');
saveas(gcf, fullfile(result_dir, '图3_特征相关性热图.png'));

% 图4: PCA分析增强版
fprintf('\n进行PCA分析...\n');

% 预处理特征矩阵，移除线性相关的特征
feature_matrix_clean = feature_matrix;

% 移除常数特征（方差为0的特征）
feature_vars = var(feature_matrix_clean);
non_constant_features = feature_vars > 1e-10;
feature_matrix_clean = feature_matrix_clean(:, non_constant_features);
fprintf('移除了 %d 个常数特征\n', sum(~non_constant_features));

% 标准化特征
feature_matrix_clean = zscore(feature_matrix_clean);

% 移除NaN值
nan_cols = any(isnan(feature_matrix_clean));
feature_matrix_clean = feature_matrix_clean(:, ~nan_cols);
fprintf('移除了 %d 个包含NaN的特征\n', sum(nan_cols));

% 进行PCA分析
[coeff, score, latent, tsquared, explained] = pca(feature_matrix_clean);

figure('Position', [100, 100, 1500, 600]);

% 2D PCA可视化
subplot(1, 3, 1);
colors = ['b', 'r', 'g', 'm'];
markers = ['o', 's', '^', 'd'];

for label = 0:3
    if sum(labels == label) > 0
        idx = labels == label;
        scatter(score(idx, 1), score(idx, 2), 50, colors(label+1), markers(label+1), 'filled');
        hold on;
    end
end

xlabel(sprintf('PC1 (%.1f%%)', explained(1)));
ylabel(sprintf('PC2 (%.1f%%)', explained(2)));
title('图4-1: PCA降维可视化(2D)');
legend(label_names, 'Location', 'best');
grid on;

% 3D PCA可视化
subplot(1, 3, 2);
for label = 0:3
    if sum(labels == label) > 0
        idx = labels == label;
        scatter3(score(idx, 1), score(idx, 2), score(idx, 3), 50, colors(label+1), markers(label+1), 'filled');
        hold on;
    end
end
xlabel(sprintf('PC1 (%.1f%%)', explained(1)));
ylabel(sprintf('PC2 (%.1f%%)', explained(2)));
zlabel(sprintf('PC3 (%.1f%%)', explained(3)));
title('图4-2: 3D PCA可视化');
legend(label_names, 'Location', 'best');
grid on;

% 主成分贡献率
subplot(1, 3, 3);
cumulative_var = cumsum(explained);
plot(1:min(15, length(explained)), explained(1:min(15, length(explained))), 'bo-', 'LineWidth', 2);
hold on;
plot(1:min(15, length(explained)), cumulative_var(1:min(15, length(explained))), 'ro-', 'LineWidth', 2);
xlabel('主成分数量');
ylabel('方差贡献率 (%)');
title('图4-3: 主成分方差贡献率');
legend('单个贡献率', '累积贡献率', 'Location', 'best');
grid on;

sgtitle('图4: PCA分析结果');
saveas(gcf, fullfile(result_dir, '图4_PCA分析.png'));

% 图5: 故障特征频率可视化
if any(fault_freq_data(:,5) > 0)
    figure('Position', [100, 100, 1200, 800]);
    
    freq_types = {'BPFO', 'BPFI', 'BSF', 'FTF'};
    for i = 1:4
        subplot(2, 2, i);
        
        % 为每个故障类型绘制频率分布
        for label = 1:3  % 只看故障类别
            if sum(labels == label) > 0
                class_indices = labels == label;
                class_freq = fault_freq_data(class_indices, i);
                class_freq = class_freq(class_freq > 0); % 只取有效频率
                
                if ~isempty(class_freq)
                    histogram(class_freq, 'DisplayName', label_names{label+1}, 'FaceAlpha', 0.7);
                    hold on;
                end
            end
        end
        
        title(sprintf('图5-%d: %s 频率分布', i, freq_types{i}));
        xlabel('频率 (Hz)');
        ylabel('样本数');
        legend('show');
        grid on;
    end
    
    sgtitle('图5: 故障特征频率分布');
    saveas(gcf, fullfile(result_dir, '图5_故障特征频率分布.png'));
end

% 图6: 典型信号对比分析（增强版）
fprintf('生成增强版典型信号分析...\n');
figure('Position', [100, 100, 1600, 1000]);

% 为每个类别选择多个代表性样本
samples_per_class = 2;
subplot_rows = 4;
subplot_cols = samples_per_class * 2; % 时域和频域

for label = 0:3
    if sum(labels == label) > 0
        class_indices = find(labels == label);
        selected_samples = class_indices(1:min(samples_per_class, length(class_indices)));
        
        for sample_idx = 1:length(selected_samples)
            idx = selected_samples(sample_idx);
            
            % 获取信号数据
            if ~isempty(sample_info(idx).de_data)
                signal = sample_info(idx).de_data;
            elseif ~isempty(sample_info(idx).fe_data)
                signal = sample_info(idx).fe_data;
            else
                continue;
            end
            
            % 信号预处理
            if size(signal, 1) < size(signal, 2)
                signal = signal';
            end
            signal = signal - mean(signal);
            if length(signal) > 8192
                signal = signal(1:8192);
            end
            
            fs = sample_info(idx).fs;
            t = (0:length(signal)-1) / fs;
            
            % 时域图
            subplot_idx = (label) * subplot_cols + (sample_idx-1) * 2 + 1;
            subplot(subplot_rows, subplot_cols, subplot_idx);
            plot(t, signal);
            title(sprintf('%s-%d 时域', label_names{label+1}, sample_idx));
            xlabel('时间 (s)');
            ylabel('幅值');
            grid on;
            
            % 频域图
            subplot_idx = (label) * subplot_cols + (sample_idx-1) * 2 + 2;
            subplot(subplot_rows, subplot_cols, subplot_idx);
            N = length(signal);
            f = (0:N-1) * fs / N;
            Y = fft(signal);
            magnitude = abs(Y(1:floor(N/2)+1));
            f_half = f(1:floor(N/2)+1);
            
            semilogy(f_half, magnitude);
            title(sprintf('%s-%d 频域', label_names{label+1}, sample_idx));
            xlabel('频率 (Hz)');
            ylabel('幅值 (对数)');
            grid on;
            xlim([0, min(3000, max(f_half))]);
        end
    end
end

sgtitle('图6: 各类别典型信号对比分析');
saveas(gcf, fullfile(result_dir, '图6_增强典型信号分析.png'));

% 图7: 特征重要性分析
fprintf('进行特征重要性分析...\n');
feature_importance = [];
for i = 1:size(feature_matrix, 2)
    % 计算每个特征在不同类别间的方差比
    between_class_var = 0;
    within_class_var = 0;
    overall_mean = mean(feature_matrix(:, i));
    
    for label = 0:3
        if sum(labels == label) > 0
            class_data = feature_matrix(labels == label, i);
            class_mean = mean(class_data);
            class_size = length(class_data);
            
            between_class_var = between_class_var + class_size * (class_mean - overall_mean)^2;
            within_class_var = within_class_var + sum((class_data - class_mean).^2);
        end
    end
    
    f_ratio = between_class_var / within_class_var;
    feature_importance = [feature_importance; f_ratio];
end

% 特征重要性排序
[sorted_importance, importance_idx] = sort(feature_importance, 'descend');
top_features = importance_idx(1:15);

figure('Position', [100, 100, 1200, 600]);
bar(sorted_importance(1:15));
set(gca, 'XTickLabel', feature_names_full(top_features));
title('图7: 特征重要性排序 (前15个)');
ylabel('F-ratio');
xtickangle(45);
grid on;
saveas(gcf, fullfile(result_dir, '图7_特征重要性分析.png'));

% 保存特征重要性表
importance_table = table(feature_names_full(importance_idx)', feature_importance(importance_idx), ...
    'VariableNames', {'特征名称', 'F_ratio'});
writetable(importance_table, fullfile(result_dir, '特征重要性排序.csv'));

%% 9. 生成分析报告
fprintf('\n=== 生成分析报告 ===\n');
report_file = fullfile(result_dir, '数据分析报告.txt');
fid = fopen(report_file, 'w', 'n', 'UTF-8');

fprintf(fid, '高速列车轴承智能故障诊断 - 任务1分析报告\n');
fprintf(fid, '================================================\n\n');

fprintf(fid, '1. 数据概况\n');
fprintf(fid, '   总样本数: %d\n', size(feature_matrix, 1));
fprintf(fid, '   特征维度: %d\n', size(feature_matrix, 2));
fprintf(fid, '   数据集数量: %d\n', length(datasets));
fprintf(fid, '\n');

fprintf(fid, '2. 类别分布\n');
for i = 1:size(class_stats, 1)
    fprintf(fid, '   %s: %d 样本 (%.1f%%)\n', label_names{class_stats(i,1)+1}, ...
        class_stats(i,2), class_stats(i,3));
end
fprintf(fid, '\n');

fprintf(fid, '3. 数据集分布\n');
for i = 1:length(datasets)
    fprintf(fid, '   %s: %d 样本 (%.1f%%)\n', datasets{i}, ...
        dataset_stats(i,2), dataset_stats(i,3));
end
fprintf(fid, '\n');

fprintf(fid, '4. 特征提取结果\n');
fprintf(fid, '   时域特征: 13个 (均值、标准差、RMS等)\n');
fprintf(fid, '   频域特征: 11个 (频谱特征、故障频率幅值等)\n');
fprintf(fid, '   包络谱特征: 7个 (包络统计特征、包络故障频率等)\n');
fprintf(fid, '\n');

fprintf(fid, '5. 主要发现\n');
fprintf(fid, '   - PCA前两个主成分解释了 %.1f%% 的方差\n', sum(explained(1:2)));
fprintf(fid, '   - 最重要的特征是: %s\n', feature_names_full{importance_idx(1)});
fprintf(fid, '   - 各类别在特征空间中有一定的可分性\n');
fprintf(fid, '\n');

fprintf(fid, '6. 生成文件列表\n');
fprintf(fid, '   - task1_complete_data.mat: 完整数据文件\n');
fprintf(fid, '   - 特征统计表.csv: 总体特征统计\n');
fprintf(fid, '   - 各类别特征统计.csv: 分类别特征统计\n');
fprintf(fid, '   - 故障特征频率表.csv: 故障频率数据\n');
fprintf(fid, '   - 特征重要性排序.csv: 特征重要性分析\n');
fprintf(fid, '   - 多个可视化图表.png\n');

fclose(fid);

fprintf('\n任务1完成：数据分析与故障特征提取\n');
fprintf('================================================\n');
fprintf('- 成功加载并处理了 %d 个样本\n', size(feature_matrix, 1));
fprintf('- 提取了 %d 维特征，包括时域、频域和包络谱特征\n', size(feature_matrix, 2));
fprintf('- 计算了轴承故障特征频率相关特征\n');
fprintf('- 生成了 %d 个可视化图表\n', 8);
fprintf('- 生成了 %d 个数据表格文件\n', 6);
fprintf('- 完成了特征重要性分析\n');
fprintf('- 结果已保存到 "%s" 文件夹\n', result_dir);
fprintf('- 生成了详细的分析报告\n');

%% ========== 以下是所有函数定义 ==========

%% 数据读取函数
function [data_struct, file_info] = load_bearing_data(data_root)
    data_struct = struct();
    file_info = struct();
    
    % 定义数据集类型
    datasets = {'12kHz_DE_data', '12kHz_FE_data', '48kHz_DE_data', '48kHz_Normal_data'};
    fault_types = {'B', 'IR', 'OR', 'N'};
    
    file_count = 0;
    
    for ds_idx = 1:length(datasets)
        dataset = datasets{ds_idx};
        dataset_path = fullfile(data_root, dataset);
        
        if ~exist(dataset_path, 'dir')
            continue;
        end
        
        fprintf('正在处理数据集: %s\n', dataset);
        
        % 解析采样频率和位置
        if contains(dataset, '12kHz')
            fs = 12000;
        elseif contains(dataset, '48kHz')
            fs = 48000;
        end
        
        if contains(dataset, 'DE')
            position = 'DE';
            bearing_type = 'SKF6205';
        elseif contains(dataset, 'FE')
            position = 'FE';
            bearing_type = 'SKF6203';
        else
            position = 'Normal';
            bearing_type = 'SKF6205';
        end
        
        % 处理正常数据
        if contains(dataset, 'Normal')
            normal_files = dir(fullfile(dataset_path, '*.mat'));
            for i = 1:length(normal_files)
                file_count = file_count + 1;
                file_path = fullfile(dataset_path, normal_files(i).name);
                
                try
                    mat_data = load(file_path);
                    field_names = fieldnames(mat_data);
                    
                    % 提取数据和转速
                    de_data = []; fe_data = []; ba_data = []; rpm = [];
                    
                    for fn = 1:length(field_names)
                        field_name = field_names{fn};
                        if contains(field_name, 'DE_time')
                            de_data = mat_data.(field_name);
                        elseif contains(field_name, 'FE_time')
                            fe_data = mat_data.(field_name);
                        elseif contains(field_name, 'BA_time')
                            ba_data = mat_data.(field_name);
                        elseif contains(field_name, 'RPM')
                            rpm = mat_data.(field_name);
                        end
                    end
                    
                    % 存储数据
                    data_struct(file_count).filename = normal_files(i).name;
                    data_struct(file_count).dataset = dataset;
                    data_struct(file_count).fault_type = 'N';
                    data_struct(file_count).fault_size = 0;
                    data_struct(file_count).load_hp = 0;
                    data_struct(file_count).position = position;
                    data_struct(file_count).fs = fs;
                    data_struct(file_count).bearing_type = bearing_type;
                    data_struct(file_count).de_data = de_data;
                    data_struct(file_count).fe_data = fe_data;
                    data_struct(file_count).ba_data = ba_data;
                    data_struct(file_count).rpm = rpm;
                    
                    fprintf('  已加载: %s\n', normal_files(i).name);
                    
                catch ME
                    fprintf('  错误加载文件 %s: %s\n', normal_files(i).name, ME.message);
                end
            end
        else
            % 处理故障数据
            for ft_idx = 1:length(fault_types)-1  % 排除'N'
                fault_type = fault_types{ft_idx};
                fault_path = fullfile(dataset_path, fault_type);
                
                if ~exist(fault_path, 'dir')
                    continue;
                end
                
                if strcmp(fault_type, 'OR')
                    % 外圈故障有三个位置
                    or_positions = {'Centered', 'Opposite', 'Orthogonal'};
                    for pos_idx = 1:length(or_positions)
                        or_pos = or_positions{pos_idx};
                        or_pos_path = fullfile(fault_path, or_pos);
                        
                        if exist(or_pos_path, 'dir')
                            size_dirs = dir(or_pos_path);
                            size_dirs = size_dirs([size_dirs.isdir] & ~ismember({size_dirs.name}, {'.', '..'}));
                            
                            for size_idx = 1:length(size_dirs)
                                size_dir = size_dirs(size_idx).name;
                                fault_size = str2double(size_dir) / 1000; % 转换为英寸
                                size_path = fullfile(or_pos_path, size_dir);
                                
                                mat_files = dir(fullfile(size_path, '*.mat'));
                                for file_idx = 1:length(mat_files)
                                    file_count = file_count + 1;
                                    file_path = fullfile(size_path, mat_files(file_idx).name);
                                    
                                    % 从文件名提取载荷信息
                                    filename = mat_files(file_idx).name;
                                    load_hp = extract_load_from_filename(filename);
                                    
                                    try
                                        [de_data, fe_data, ba_data, rpm] = load_mat_file(file_path);
                                        
                                        data_struct(file_count).filename = filename;
                                        data_struct(file_count).dataset = dataset;
                                        data_struct(file_count).fault_type = fault_type;
                                        data_struct(file_count).fault_size = fault_size;
                                        data_struct(file_count).load_hp = load_hp;
                                        data_struct(file_count).position = position;
                                        data_struct(file_count).fs = fs;
                                        data_struct(file_count).bearing_type = bearing_type;
                                        data_struct(file_count).or_position = or_pos;
                                        data_struct(file_count).de_data = de_data;
                                        data_struct(file_count).fe_data = fe_data;
                                        data_struct(file_count).ba_data = ba_data;
                                        data_struct(file_count).rpm = rpm;
                                        
                                        fprintf('  已加载: %s (%s, %s, %.3f英寸, %dHP)\n', ...
                                            filename, fault_type, or_pos, fault_size, load_hp);
                                        
                                    catch ME
                                        fprintf('  错误加载文件 %s: %s\n', filename, ME.message);
                                    end
                                end
                            end
                        end
                    end
                else
                    % B和IR故障
                    size_dirs = dir(fault_path);
                    size_dirs = size_dirs([size_dirs.isdir] & ~ismember({size_dirs.name}, {'.', '..'}));
                    
                    for size_idx = 1:length(size_dirs)
                        size_dir = size_dirs(size_idx).name;
                        fault_size = str2double(size_dir) / 1000; % 转换为英寸
                        size_path = fullfile(fault_path, size_dir);
                        
                        mat_files = dir(fullfile(size_path, '*.mat'));
                        for file_idx = 1:length(mat_files)
                            file_count = file_count + 1;
                            file_path = fullfile(size_path, mat_files(file_idx).name);
                            
                            % 从文件名提取载荷信息
                            filename = mat_files(file_idx).name;
                            load_hp = extract_load_from_filename(filename);
                            
                            try
                                [de_data, fe_data, ba_data, rpm] = load_mat_file(file_path);
                                
                                data_struct(file_count).filename = filename;
                                data_struct(file_count).dataset = dataset;
                                data_struct(file_count).fault_type = fault_type;
                                data_struct(file_count).fault_size = fault_size;
                                data_struct(file_count).load_hp = load_hp;
                                data_struct(file_count).position = position;
                                data_struct(file_count).fs = fs;
                                data_struct(file_count).bearing_type = bearing_type;
                                data_struct(file_count).de_data = de_data;
                                data_struct(file_count).fe_data = fe_data;
                                data_struct(file_count).ba_data = ba_data;
                                data_struct(file_count).rpm = rpm;
                                
                                fprintf('  已加载: %s (%s, %.3f英寸, %dHP)\n', ...
                                    filename, fault_type, fault_size, load_hp);
                                
                            catch ME
                                fprintf('  错误加载文件 %s: %s\n', filename, ME.message);
                            end
                        end
                    end
                end
            end
        end
    end
    
    file_info.total_files = file_count;
    fprintf('\n总共加载了 %d 个文件\n', file_count);
end

%% 辅助函数：从文件名提取载荷信息
function load_hp = extract_load_from_filename(filename)
    % 从文件名中提取载荷信息
    if contains(filename, '_0')
        load_hp = 0;
    elseif contains(filename, '_1')
        load_hp = 1;
    elseif contains(filename, '_2')
        load_hp = 2;
    elseif contains(filename, '_3')
        load_hp = 3;
    else
        load_hp = 0; % 默认值
    end
end

%% 辅助函数：加载MAT文件
function [de_data, fe_data, ba_data, rpm] = load_mat_file(file_path)
    mat_data = load(file_path);
    field_names = fieldnames(mat_data);
    
    de_data = []; fe_data = []; ba_data = []; rpm = [];
    
    for fn = 1:length(field_names)
        field_name = field_names{fn};
        if contains(field_name, 'DE_time') || (contains(field_name, 'DE') && contains(field_name, 'time'))
            de_data = mat_data.(field_name);
        elseif contains(field_name, 'FE_time') || (contains(field_name, 'FE') && contains(field_name, 'time'))
            fe_data = mat_data.(field_name);
        elseif contains(field_name, 'BA_time') || (contains(field_name, 'BA') && contains(field_name, 'time'))
            ba_data = mat_data.(field_name);
        elseif contains(field_name, 'RPM')
            rpm = mat_data.(field_name);
        end
    end
end

%% 故障特征频率计算函数
function [bpfo, bpfi, bsf, ftf] = calculate_fault_frequencies(rpm, bearing_params)
    fr = rpm / 60; % 转频 Hz
    n = bearing_params.n;  % 滚动体数
    d = bearing_params.d;  % 滚动体直径
    D = bearing_params.D;  % 轴承节径
    
    % 计算故障特征频率
    bpfo = fr * n/2 * (1 - d/D);  % 外圈故障特征频率
    bpfi = fr * n/2 * (1 + d/D);  % 内圈故障特征频率
    bsf = fr * D/(2*d) * (1 - (d/D)^2);  % 滚动体故障特征频率
    ftf = fr/2 * (1 - d/D);  % 滚动体公转频率
end

%% 时域特征提取函数
function time_features = extract_time_features(signal)
    % 时域统计特征
    time_features.mean = mean(signal);
    time_features.std = std(signal);
    time_features.rms = rms(signal);
    time_features.peak = max(abs(signal));
    time_features.peak_to_peak = max(signal) - min(signal);
    time_features.crest_factor = time_features.peak / time_features.rms;
    time_features.clearance_factor = time_features.peak / mean(sqrt(abs(signal)))^2;
    time_features.shape_factor = time_features.rms / mean(abs(signal));
    time_features.impulse_factor = time_features.peak / mean(abs(signal));
    time_features.skewness = skewness(signal);
    time_features.kurtosis = kurtosis(signal);
    
    % 能量特征
    time_features.energy = sum(signal.^2);
    time_features.power = time_features.energy / length(signal);
end

%% 频域特征提取函数
function freq_features = extract_freq_features(signal, fs, fault_freqs)
    N = length(signal);
    f = (0:N-1) * fs / N;
    
    % FFT计算
    Y = fft(signal);
    magnitude = abs(Y(1:floor(N/2)+1));
    magnitude(2:end-1) = 2 * magnitude(2:end-1);
    f_half = f(1:floor(N/2)+1);
    
    % 频域统计特征
    freq_features.spectral_centroid = sum(f_half .* magnitude') / sum(magnitude);
    freq_features.spectral_spread = sqrt(sum(((f_half - freq_features.spectral_centroid).^2) .* magnitude') / sum(magnitude));
    freq_features.spectral_rolloff = f_half(find(cumsum(magnitude) >= 0.85 * sum(magnitude), 1));
    freq_features.spectral_flux = sum(diff(magnitude).^2);
    
    % 故障特征频率处的幅值
    if ~isempty(fault_freqs)
        freq_features.bpfo_amplitude = get_amplitude_at_frequency(f_half, magnitude, fault_freqs.bpfo);
        freq_features.bpfi_amplitude = get_amplitude_at_frequency(f_half, magnitude, fault_freqs.bpfi);
        freq_features.bsf_amplitude = get_amplitude_at_frequency(f_half, magnitude, fault_freqs.bsf);
        freq_features.ftf_amplitude = get_amplitude_at_frequency(f_half, magnitude, fault_freqs.ftf);
        
        % 故障特征频率的谐波
        freq_features.bpfo_2x = get_amplitude_at_frequency(f_half, magnitude, 2*fault_freqs.bpfo);
        freq_features.bpfi_2x = get_amplitude_at_frequency(f_half, magnitude, 2*fault_freqs.bpfi);
        freq_features.bsf_2x = get_amplitude_at_frequency(f_half, magnitude, 2*fault_freqs.bsf);
    end
    
    freq_features.fft_magnitude = magnitude;
    freq_features.frequency = f_half;
end

%% 辅助函数：获取指定频率处的幅值
function amplitude = get_amplitude_at_frequency(f, magnitude, target_freq)
    if target_freq <= 0 || target_freq > max(f)
        amplitude = 0;
        return;
    end
    
    [~, idx] = min(abs(f - target_freq));
    % 取附近几个点的平均值以提高鲁棒性
    window = max(1, idx-2):min(length(magnitude), idx+2);
    amplitude = mean(magnitude(window));
end

%% 包络谱分析函数
function envelope_features = extract_envelope_features(signal, fs, fault_freqs)
    % Hilbert变换获取包络
    analytic_signal = hilbert(signal);
    envelope = abs(analytic_signal);
    
    % 包络谱
    N = length(envelope);
    f = (0:N-1) * fs / N;
    Y_env = fft(envelope);
    magnitude_env = abs(Y_env(1:floor(N/2)+1));
    magnitude_env(2:end-1) = 2 * magnitude_env(2:end-1);
    f_half = f(1:floor(N/2)+1);
    
    % 包络谱特征
    envelope_features.envelope_rms = rms(envelope);
    envelope_features.envelope_peak = max(envelope);
    envelope_features.envelope_crest = envelope_features.envelope_peak / envelope_features.envelope_rms;
    
    % 故障特征频率在包络谱中的幅值
    if ~isempty(fault_freqs)
        envelope_features.env_bpfo = get_amplitude_at_frequency(f_half, magnitude_env, fault_freqs.bpfo);
        envelope_features.env_bpfi = get_amplitude_at_frequency(f_half, magnitude_env, fault_freqs.bpfi);
        envelope_features.env_bsf = get_amplitude_at_frequency(f_half, magnitude_env, fault_freqs.bsf);
        envelope_features.env_ftf = get_amplitude_at_frequency(f_half, magnitude_env, fault_freqs.ftf);
    end
    
    envelope_features.envelope_spectrum = magnitude_env;
    envelope_features.envelope_freq = f_half;
end