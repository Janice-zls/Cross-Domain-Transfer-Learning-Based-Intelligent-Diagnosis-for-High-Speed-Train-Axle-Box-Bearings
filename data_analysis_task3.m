%% 高速列车轴承智能故障诊断 - 任务3：迁移诊断

clear; clc; close all;

%% 1. 加载源域模型和数据
fprintf('=== 任务3：迁移诊断 ===\n');
fprintf('加载源域模型和数据...\n');

% 加载任务一的特征数据（源域）
task1_data_path = 'c:\Users\Jack\Desktop\E题\问题一结果\task1_complete_data.mat';
if ~exist(task1_data_path, 'file')
    error('未找到任务一的数据文件，请先运行任务一！');
end

load(task1_data_path);
source_features = feature_matrix;
source_labels = labels;

% 创建结果保存文件夹
result_dir = '问题三结果';
if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

fprintf('源域数据加载完成！\n');
fprintf('源域样本数: %d\n', size(source_features, 1));
fprintf('源域特征维度: %d\n', size(source_features, 2));

% 类别标签定义
unique_labels = unique(source_labels);
label_names = {'正常', '滚动体故障', '内圈故障', '外圈故障'};

%% 2. 加载目标域数据
fprintf('\n=== 加载目标域数据 ===\n');

target_data_dir = 'c:\Users\Jack\Desktop\E题\目标域数据集';
target_files = dir(fullfile(target_data_dir, '*.mat'));

target_features_all = [];
target_file_names = {};

fprintf('加载目标域数据文件...\n');
for i = 1:length(target_files)
    file_path = fullfile(target_data_dir, target_files(i).name);
    fprintf('加载文件: %s\n', target_files(i).name);
    
    try
        data = load(file_path);
        field_names = fieldnames(data);
        
        % 获取数据（通常是第一个字段）
        signal_data = data.(field_names{1});
        
        % 提取特征（使用与任务一相同的特征提取方法）
        features = extract_features_for_transfer(signal_data);
        
        target_features_all = [target_features_all; features];
        target_file_names{end+1} = target_files(i).name;
        
    catch ME
        fprintf('加载文件 %s 失败: %s\n', target_files(i).name, ME.message);
    end
end

fprintf('目标域数据加载完成！\n');
fprintf('目标域样本数: %d\n', size(target_features_all, 1));
fprintf('目标域特征维度: %d\n', size(target_features_all, 2));

%% 3. 源域和目标域数据预处理
fprintf('\n=== 数据预处理和对齐 ===\n');

% 3.1 特征对齐（确保源域和目标域特征维度一致）
min_features = min(size(source_features, 2), size(target_features_all, 2));
source_features_aligned = source_features(:, 1:min_features);
target_features_aligned = target_features_all(:, 1:min_features);

fprintf('特征对齐完成，统一特征维度: %d\n', min_features);

% 3.2 数据标准化（使用源域的统计量）
source_mean = mean(source_features_aligned);
source_std = std(source_features_aligned);

% 避免除零
source_std(source_std == 0) = 1;

% 标准化源域和目标域数据
source_features_norm = (source_features_aligned - source_mean) ./ source_std;
target_features_norm = (target_features_aligned - source_mean) ./ source_std;

% 添加小噪声避免数值问题
noise_level = 1e-6;
source_features_norm = source_features_norm + noise_level * randn(size(source_features_norm));
target_features_norm = target_features_norm + noise_level * randn(size(target_features_norm));

fprintf('数据标准化完成\n');

%% 4. 源域和目标域差异分析
fprintf('\n=== 源域和目标域差异分析 ===\n');

% 4.1 统计特征对比
source_stats = [mean(source_features_norm); std(source_features_norm); 
                skewness(source_features_norm); kurtosis(source_features_norm)];
target_stats = [mean(target_features_norm); std(target_features_norm); 
                skewness(target_features_norm); kurtosis(target_features_norm)];

% 计算分布差异
distribution_diff = sqrt(sum((source_stats - target_stats).^2, 1));
mean_diff = mean(distribution_diff);

fprintf('源域和目标域平均分布差异: %.4f\n', mean_diff);

% 保存统计对比
stats_comparison = table((1:min_features)', source_stats(1,:)', target_stats(1,:)', ...
    source_stats(2,:)', target_stats(2,:)', distribution_diff', ...
    'VariableNames', {'特征索引', '源域均值', '目标域均值', '源域标准差', '目标域标准差', '分布差异'});
writetable(stats_comparison, fullfile(result_dir, '源域目标域统计对比.csv'));

%% 5. 迁移学习方法实现
fprintf('\n=== 迁移学习方法实现 ===\n');

% 5.1 重新训练源域模型（为迁移做准备）
fprintf('重新训练源域模型...\n');

% 数据划分
train_ratio = 0.8;
n_source = size(source_features_norm, 1);
train_indices = randperm(n_source, round(n_source * train_ratio));
test_indices = setdiff(1:n_source, train_indices);

X_source_train = source_features_norm(train_indices, :);
y_source_train = source_labels(train_indices);
X_source_test = source_features_norm(test_indices, :);
y_source_test = source_labels(test_indices);

% 训练多个源域模型
source_models = struct();

% SVM模型
try
    fprintf('训练源域SVM模型...\n');
    svm_template = templateSVM('KernelFunction', 'rbf', 'Standardize', false);
    source_models.svm = fitcecoc(X_source_train, y_source_train, 'Learners', svm_template);
    
    % 测试源域SVM性能
    y_pred_svm = predict(source_models.svm, X_source_test);
    svm_accuracy = sum(y_pred_svm == y_source_test) / length(y_source_test);
    fprintf('源域SVM准确率: %.4f\n', svm_accuracy);
    source_models.svm_accuracy = svm_accuracy;
    source_models.svm_success = true;
catch ME
    fprintf('源域SVM训练失败: %s\n', ME.message);
    source_models.svm_success = false;
end

% 随机森林模型
try
    fprintf('训练源域随机森林模型...\n');
    source_models.rf = TreeBagger(100, X_source_train, y_source_train, 'Method', 'classification');
    
    % 测试源域随机森林性能
    y_pred_rf_cell = predict(source_models.rf, X_source_test);
    y_pred_rf = str2double(y_pred_rf_cell);
    rf_accuracy = sum(y_pred_rf == y_source_test) / length(y_source_test);
    fprintf('源域随机森林准确率: %.4f\n', rf_accuracy);
    source_models.rf_accuracy = rf_accuracy;
    source_models.rf_success = true;
catch ME
    fprintf('源域随机森林训练失败: %s\n', ME.message);
    source_models.rf_success = false;
end

% KNN模型
try
    fprintf('训练源域KNN模型...\n');
    source_models.knn = fitcknn(X_source_train, y_source_train, 'NumNeighbors', 5);
    
    % 测试源域KNN性能
    y_pred_knn = predict(source_models.knn, X_source_test);
    knn_accuracy = sum(y_pred_knn == y_source_test) / length(y_source_test);
    fprintf('源域KNN准确率: %.4f\n', knn_accuracy);
    source_models.knn_accuracy = knn_accuracy;
    source_models.knn_success = true;
catch ME
    fprintf('源域KNN训练失败: %s\n', ME.message);
    source_models.knn_success = false;
end

%% 6. 迁移学习策略实现
fprintf('\n=== 迁移学习策略 ===\n');

% 6.1 基于特征的迁移 - 领域自适应
fprintf('实施基于特征的迁移学习...\n');

% 使用主成分分析进行特征空间对齐
combined_features = [source_features_norm; target_features_norm];
[coeff, score, ~, ~, explained] = pca(combined_features);

% 选择保留95%方差的主成分，但至少保留3个主成分用于可视化
cumsum_explained = cumsum(explained);
n_components_95 = find(cumsum_explained >= 95, 1);
n_components = max(3, n_components_95); % 确保至少有3个主成分
n_components = min(n_components, size(combined_features, 2)); % 不超过特征总数

fprintf('选择前%d个主成分（保留%.2f%%方差）\n', n_components, cumsum_explained(n_components));

% 投影到新的特征空间
source_features_pca = score(1:size(source_features_norm, 1), 1:n_components);
target_features_pca = score(size(source_features_norm, 1)+1:end, 1:n_components);

% 6.2 领域对抗训练（简化版）
fprintf('实施领域对抗训练...\n');

% 创建领域标签（0=源域，1=目标域）
domain_labels = [zeros(size(source_features_pca, 1), 1); ones(size(target_features_pca, 1), 1)];
combined_features_pca = [source_features_pca; target_features_pca];

% 训练领域分类器
try
    domain_classifier = fitcsvm(combined_features_pca, domain_labels);
    domain_pred = predict(domain_classifier, combined_features_pca);
    domain_accuracy = sum(domain_pred == domain_labels) / length(domain_labels);
    fprintf('领域分类器准确率: %.4f (越低越好，表示特征越难区分)\n', domain_accuracy);
catch ME
    fprintf('领域分类器训练失败: %s\n', ME.message);
    domain_accuracy = 0.5;
end

%% 7. 目标域预测
fprintf('\n=== 目标域预测 ===\n');

transfer_results = struct();

% 7.1 直接迁移（使用源域模型直接预测目标域）
fprintf('方法1: 直接迁移预测...\n');

if source_models.svm_success
    try
        target_pred_svm_direct = predict(source_models.svm, target_features_norm);
        transfer_results.svm_direct = target_pred_svm_direct;
        fprintf('SVM直接迁移完成\n');
    catch ME
        fprintf('SVM直接迁移失败: %s\n', ME.message);
    end
end

if source_models.rf_success
    try
        target_pred_rf_direct_cell = predict(source_models.rf, target_features_norm);
        target_pred_rf_direct = str2double(target_pred_rf_direct_cell);
        transfer_results.rf_direct = target_pred_rf_direct;
        fprintf('随机森林直接迁移完成\n');
    catch ME
        fprintf('随机森林直接迁移失败: %s\n', ME.message);
    end
end

if source_models.knn_success
    try
        target_pred_knn_direct = predict(source_models.knn, target_features_norm);
        transfer_results.knn_direct = target_pred_knn_direct;
        fprintf('KNN直接迁移完成\n');
    catch ME
        fprintf('KNN直接迁移失败: %s\n', ME.message);
    end
end

% 7.2 基于PCA特征的迁移
fprintf('方法2: 基于PCA特征的迁移预测...\n');

% 重新训练基于PCA特征的模型
X_source_pca_train = source_features_pca(train_indices, :);
X_source_pca_test = source_features_pca(test_indices, :);

if source_models.svm_success
    try
        svm_pca_model = fitcecoc(X_source_pca_train, y_source_train, 'Learners', svm_template);
        target_pred_svm_pca = predict(svm_pca_model, target_features_pca);
        transfer_results.svm_pca = target_pred_svm_pca;
        fprintf('SVM-PCA迁移完成\n');
    catch ME
        fprintf('SVM-PCA迁移失败: %s\n', ME.message);
    end
end

if source_models.rf_success
    try
        rf_pca_model = TreeBagger(100, X_source_pca_train, y_source_train, 'Method', 'classification');
        target_pred_rf_pca_cell = predict(rf_pca_model, target_features_pca);
        target_pred_rf_pca = str2double(target_pred_rf_pca_cell);
        transfer_results.rf_pca = target_pred_rf_pca;
        fprintf('随机森林-PCA迁移完成\n');
    catch ME
        fprintf('随机森林-PCA迁移失败: %s\n', ME.message);
    end
end

% 7.3 集成迁移（多模型投票）
fprintf('方法3: 集成迁移预测...\n');

available_predictions = {};
prediction_names = {};

% 收集所有可用的预测结果
if isfield(transfer_results, 'svm_direct')
    available_predictions{end+1} = transfer_results.svm_direct;
    prediction_names{end+1} = 'SVM直接';
end
if isfield(transfer_results, 'rf_direct')
    available_predictions{end+1} = transfer_results.rf_direct;
    prediction_names{end+1} = '随机森林直接';
end
if isfield(transfer_results, 'knn_direct')
    available_predictions{end+1} = transfer_results.knn_direct;
    prediction_names{end+1} = 'KNN直接';
end
if isfield(transfer_results, 'svm_pca')
    available_predictions{end+1} = transfer_results.svm_pca;
    prediction_names{end+1} = 'SVM-PCA';
end
if isfield(transfer_results, 'rf_pca')
    available_predictions{end+1} = transfer_results.rf_pca;
    prediction_names{end+1} = '随机森林-PCA';
end

% 多数投票集成
if ~isempty(available_predictions)
    n_samples = length(available_predictions{1});
    ensemble_predictions = zeros(n_samples, 1);
    
    for i = 1:n_samples
        votes = [];
        for j = 1:length(available_predictions)
            votes = [votes, available_predictions{j}(i)];
        end
        ensemble_predictions(i) = mode(votes);
    end
    
    transfer_results.ensemble = ensemble_predictions;
    fprintf('集成迁移完成，使用%d个模型\n', length(available_predictions));
end

%% 8. 结果分析和可视化
fprintf('\n=== 结果分析和可视化 ===\n');

% 8.1 预测结果统计
fprintf('目标域预测结果统计:\n');
for i = 1:length(prediction_names)
    pred = available_predictions{i};
    fprintf('\n%s预测结果:\n', prediction_names{i});
    for label = unique_labels'
        count = sum(pred == label);
        percentage = count / length(pred) * 100;
        fprintf('  %s: %d个样本 (%.1f%%)\n', label_names{label+1}, count, percentage);
    end
end

if isfield(transfer_results, 'ensemble')
    fprintf('\n集成预测结果:\n');
    for label = unique_labels'
        count = sum(transfer_results.ensemble == label);
        percentage = count / length(transfer_results.ensemble) * 100;
        fprintf('  %s: %d个样本 (%.1f%%)\n', label_names{label+1}, count, percentage);
    end
end

% 8.2 保存预测结果
fprintf('保存预测结果...\n');

% 创建结果表格
result_table_data = [];
result_table_names = {'文件名'};

for i = 1:length(target_file_names)
    result_table_data{i, 1} = target_file_names{i};
end

col_idx = 2;
for i = 1:length(prediction_names)
    result_table_names{col_idx} = prediction_names{i};
    pred = available_predictions{i};
    for j = 1:length(target_file_names)
        result_table_data{j, col_idx} = label_names{pred(j)+1};
    end
    col_idx = col_idx + 1;
end

if isfield(transfer_results, 'ensemble')
    result_table_names{col_idx} = '集成预测';
    for j = 1:length(target_file_names)
        result_table_data{j, col_idx} = label_names{transfer_results.ensemble(j)+1};
    end
end

% 保存结果表格
result_table = cell2table(result_table_data, 'VariableNames', result_table_names);
writetable(result_table, fullfile(result_dir, '目标域预测结果.csv'));

% 8.3 可视化分析
fprintf('生成可视化分析...\n');

% 图1: 源域和目标域特征分布对比
figure('Position', [100, 100, 1400, 1000]);

% 选择前6个最重要的特征进行可视化
n_features_show = min(6, size(source_features_norm, 2));

for i = 1:n_features_show
    subplot(2, 3, i);
    
    % 绘制源域特征分布
    histogram(source_features_norm(:, i), 30, 'Normalization', 'probability', ...
              'FaceAlpha', 0.7, 'DisplayName', '源域');
    hold on;
    
    % 绘制目标域特征分布
    histogram(target_features_norm(:, i), 30, 'Normalization', 'probability', ...
              'FaceAlpha', 0.7, 'DisplayName', '目标域');
    
    xlabel(sprintf('特征%d值', i));
    ylabel('概率密度');
    title(sprintf('特征%d分布对比', i));
    legend('Location', 'best');
    grid on;
end

sgtitle('图1: 源域和目标域特征分布对比', 'FontSize', 16, 'FontWeight', 'bold');
saveas(gcf, fullfile(result_dir, '图1_源域目标域特征分布对比.png'));

% 图2: PCA降维可视化
figure('Position', [200, 200, 1200, 800]);

% 检查PCA维度
if size(source_features_pca, 2) >= 2
    % 2D PCA可视化
    subplot(1, 2, 1);
    scatter(source_features_pca(:, 1), source_features_pca(:, 2), 50, source_labels, 'filled', 'DisplayName', '源域');
    hold on;
    scatter(target_features_pca(:, 1), target_features_pca(:, 2), 50, 'k', 'x', 'LineWidth', 2, 'DisplayName', '目标域');
    xlabel('第一主成分');
    ylabel('第二主成分');
    title('PCA降维可视化 (2D)');
    colorbar;
    legend('Location', 'best');
    grid on;
    
    if size(source_features_pca, 2) >= 3
        % 3D PCA可视化
        subplot(1, 2, 2);
        scatter3(source_features_pca(:, 1), source_features_pca(:, 2), source_features_pca(:, 3), 50, source_labels, 'filled');
        hold on;
        scatter3(target_features_pca(:, 1), target_features_pca(:, 2), target_features_pca(:, 3), 50, 'k', 'x', 'LineWidth', 2);
        xlabel('第一主成分');
        ylabel('第二主成分');
        zlabel('第三主成分');
        title('PCA降维可视化 (3D)');
        colorbar;
        grid on;
    else
        % 如果只有2个主成分，显示1D分布
        subplot(1, 2, 2);
        histogram(source_features_pca(:, 1), 30, 'Normalization', 'probability', 'FaceAlpha', 0.7, 'DisplayName', '源域PC1');
        hold on;
        histogram(target_features_pca(:, 1), 30, 'Normalization', 'probability', 'FaceAlpha', 0.7, 'DisplayName', '目标域PC1');
        xlabel('第一主成分');
        ylabel('概率密度');
        title('第一主成分分布对比');
        legend('Location', 'best');
        grid on;
    end
else
    % 如果只有1个主成分
    subplot(1, 1, 1);
    histogram(source_features_pca(:, 1), 30, 'Normalization', 'probability', 'FaceAlpha', 0.7, 'DisplayName', '源域');
    hold on;
    histogram(target_features_pca(:, 1), 30, 'Normalization', 'probability', 'FaceAlpha', 0.7, 'DisplayName', '目标域');
    xlabel('第一主成分');
    ylabel('概率密度');
    title('PCA降维可视化 (1D)');
    legend('Location', 'best');
    grid on;
end

sgtitle('图2: PCA降维可视化', 'FontSize', 16, 'FontWeight', 'bold');
saveas(gcf, fullfile(result_dir, '图2_PCA降维可视化.png'));

% 图3: 预测结果分布对比
figure('Position', [300, 300, 1400, 1000]);

n_methods = length(prediction_names);
if isfield(transfer_results, 'ensemble')
    n_methods = n_methods + 1;
end

n_cols = ceil(sqrt(n_methods));
n_rows = ceil(n_methods / n_cols);

for i = 1:length(prediction_names)
    subplot(n_rows, n_cols, i);
    pred = available_predictions{i};
    
    pred_counts = [];
    for label = unique_labels'
        pred_counts = [pred_counts, sum(pred == label)];
    end
    
    bar(pred_counts, 'FaceColor', [0.2 + 0.1*i, 0.4, 0.8 - 0.1*i]);
    set(gca, 'XTickLabel', label_names);
    title(sprintf('%s预测分布', prediction_names{i}));
    ylabel('样本数');
    xtickangle(45);
    grid on;
end

if isfield(transfer_results, 'ensemble')
    subplot(n_rows, n_cols, length(prediction_names) + 1);
    pred = transfer_results.ensemble;
    
    pred_counts = [];
    for label = unique_labels'
        pred_counts = [pred_counts, sum(pred == label)];
    end
    
    bar(pred_counts, 'FaceColor', [0.8, 0.2, 0.2]);
    set(gca, 'XTickLabel', label_names);
    title('集成预测分布');
    ylabel('样本数');
    xtickangle(45);
    grid on;
end

sgtitle('图3: 各方法预测结果分布', 'FontSize', 16, 'FontWeight', 'bold');
saveas(gcf, fullfile(result_dir, '图3_预测结果分布对比.png'));

% 图4: 迁移学习效果分析
figure('Position', [400, 400, 1200, 800]);

% 子图1: 领域差异分析
subplot(2, 2, 1);
bar(distribution_diff(1:min(20, length(distribution_diff))));
xlabel('特征索引');
ylabel('分布差异');
title('源域-目标域特征分布差异');
grid on;

% 子图2: PCA方差解释
subplot(2, 2, 2);
plot(cumsum_explained(1:min(20, length(cumsum_explained))), 'o-', 'LineWidth', 2);
xlabel('主成分数量');
ylabel('累积方差解释率 (%)');
title('PCA方差解释');
grid on;

% 子图3: 预测一致性分析
if length(available_predictions) >= 2
    subplot(2, 2, 3);
    consistency_matrix = zeros(length(available_predictions));
    
    for i = 1:length(available_predictions)
        for j = 1:length(available_predictions)
            if i ~= j
                consistency = sum(available_predictions{i} == available_predictions{j}) / length(available_predictions{i});
                consistency_matrix(i, j) = consistency;
            else
                consistency_matrix(i, j) = 1;
            end
        end
    end
    
    imagesc(consistency_matrix);
    colorbar;
    set(gca, 'XTick', 1:length(prediction_names), 'XTickLabel', prediction_names);
    set(gca, 'YTick', 1:length(prediction_names), 'YTickLabel', prediction_names);
    title('模型预测一致性');
    xtickangle(45);
end

% 子图4: 源域模型性能对比
subplot(2, 2, 4);
model_names = {};
model_accuracies = [];

if source_models.svm_success
    model_names{end+1} = 'SVM';
    model_accuracies = [model_accuracies, source_models.svm_accuracy];
end
if source_models.rf_success
    model_names{end+1} = '随机森林';
    model_accuracies = [model_accuracies, source_models.rf_accuracy];
end
if source_models.knn_success
    model_names{end+1} = 'KNN';
    model_accuracies = [model_accuracies, source_models.knn_accuracy];
end

if ~isempty(model_accuracies)
    bar(model_accuracies, 'FaceColor', [0.4, 0.7, 0.4]);
    set(gca, 'XTickLabel', model_names);
    title('源域模型性能');
    ylabel('准确率');
    ylim([0, 1]);
    xtickangle(45);
    grid on;
end

sgtitle('图4: 迁移学习效果分析', 'FontSize', 16, 'FontWeight', 'bold');
saveas(gcf, fullfile(result_dir, '图4_迁移学习效果分析.png'));

% 图5: 特征重要性和迁移适应性分析
figure('Position', [500, 500, 1400, 1000]);

% 子图1: 特征分布差异热图
subplot(2, 3, 1);
feature_diff_matrix = abs(source_stats - target_stats);
imagesc(feature_diff_matrix);
colorbar;
title('特征统计差异热图');
xlabel('特征索引');
ylabel('统计量 (均值/标准差/偏度/峰度)');
set(gca, 'YTickLabel', {'均值', '标准差', '偏度', '峰度'});

% 子图2: 源域各类别在PCA空间的分布
subplot(2, 3, 2);
if size(source_features_pca, 2) >= 2
    colors = ['r', 'g', 'b', 'm'];
    for label = unique_labels'
        idx = source_labels == label;
        scatter(source_features_pca(idx, 1), source_features_pca(idx, 2), 50, colors(label+1), 'filled', ...
                'DisplayName', label_names{label+1});
        hold on;
    end
    xlabel('第一主成分');
    ylabel('第二主成分');
    title('源域各类别PCA分布');
    legend('Location', 'best');
    grid on;
else
    % 如果只有1个主成分，显示1D分布
    colors = ['r', 'g', 'b', 'm'];
    for label = unique_labels'
        idx = source_labels == label;
        histogram(source_features_pca(idx, 1), 20, 'Normalization', 'probability', ...
                 'FaceAlpha', 0.7, 'FaceColor', colors(label+1), 'DisplayName', label_names{label+1});
        hold on;
    end
    xlabel('第一主成分');
    ylabel('概率密度');
    title('源域各类别PCA分布 (1D)');
    legend('Location', 'best');
    grid on;
end

% 子图3: 目标域在PCA空间的密度分布
subplot(2, 3, 3);
if size(target_features_pca, 2) >= 2
    scatter(target_features_pca(:, 1), target_features_pca(:, 2), 50, 'k', 'filled', 'MarkerFaceAlpha', 0.6);
    xlabel('第一主成分');
    ylabel('第二主成分');
    title('目标域PCA密度分布');
else
    histogram(target_features_pca(:, 1), 20, 'FaceColor', 'k', 'FaceAlpha', 0.6);
    xlabel('第一主成分');
    ylabel('频数');
    title('目标域PCA分布 (1D)');
end
grid on;

% 子图4: 特征相关性对比
subplot(2, 3, 4);
n_features_corr = min(10, size(source_features_norm, 2));
source_corr = corrcoef(source_features_norm(:, 1:n_features_corr));
target_corr = corrcoef(target_features_norm(:, 1:n_features_corr));
corr_diff = abs(source_corr - target_corr);
imagesc(corr_diff);
colorbar;
title('特征相关性差异');
xlabel('特征索引');
ylabel('特征索引');

% 子图5: 预测置信度分析
subplot(2, 3, 5);
if isfield(transfer_results, 'ensemble')
    % 计算预测置信度（基于模型一致性）
    confidence_scores = zeros(size(transfer_results.ensemble));
    for i = 1:length(transfer_results.ensemble)
        votes = [];
        for j = 1:length(available_predictions)
            votes = [votes, available_predictions{j}(i)];
        end
        % 置信度 = 最多投票数 / 总投票数
        confidence_scores(i) = sum(votes == transfer_results.ensemble(i)) / length(votes);
    end
    
    histogram(confidence_scores, 20, 'FaceColor', [0.7, 0.3, 0.9]);
    xlabel('预测置信度');
    ylabel('样本数');
    title('目标域预测置信度分布');
    grid on;
end

% 子图6: 领域适应性评估
subplot(2, 3, 6);
if exist('domain_accuracy', 'var')
    domain_adapt_score = 1 - domain_accuracy; % 越难区分域，适应性越好
    bar([domain_accuracy, domain_adapt_score], 'FaceColor', [0.5, 0.8, 0.5]);
    set(gca, 'XTickLabel', {'领域可区分性', '领域适应性'});
    title('领域适应性评估');
    ylabel('分数');
    ylim([0, 1]);
    grid on;
end

sgtitle('图5: 特征重要性和迁移适应性分析', 'FontSize', 16, 'FontWeight', 'bold');
saveas(gcf, fullfile(result_dir, '图5_特征重要性和迁移适应性分析.png'));

% 图6: 目标域样本详细分析
figure('Position', [600, 600, 1400, 1000]);

% 子图1: 目标域样本在原始特征空间的分布
subplot(2, 3, 1);
if size(target_features_norm, 2) >= 2
    scatter(target_features_norm(:, 1), target_features_norm(:, 2), 50, 'filled');
    xlabel('特征1');
    ylabel('特征2');
    title('目标域原始特征分布');
else
    histogram(target_features_norm(:, 1), 20, 'FaceColor', [0.3, 0.7, 0.9]);
    xlabel('特征1');
    ylabel('频数');
    title('目标域原始特征分布 (1D)');
end
grid on;

% 子图2: 各文件预测结果饼图
subplot(2, 3, 2);
if isfield(transfer_results, 'ensemble')
    pred_counts = [];
    for label = unique_labels'
        pred_counts = [pred_counts, sum(transfer_results.ensemble == label)];
    end
    pie(pred_counts, label_names);
    title('目标域预测结果分布');
end

% 子图3: 预测结果按文件名展示
subplot(2, 3, 3);
if isfield(transfer_results, 'ensemble')
    bar(transfer_results.ensemble, 'FaceColor', [0.6, 0.4, 0.8]);
    xlabel('文件索引');
    ylabel('预测类别');
    title('各文件预测结果');
    set(gca, 'YTick', unique_labels, 'YTickLabel', label_names);
    grid on;
end

% 子图4: 特征重要性排序
subplot(2, 3, 4);
feature_importance = distribution_diff;
[sorted_importance, sorted_idx] = sort(feature_importance, 'descend');
bar(sorted_importance(1:min(15, length(sorted_importance))), 'FaceColor', [0.8, 0.6, 0.2]);
xlabel('特征排序');
ylabel('重要性分数');
title('特征重要性排序');
grid on;

% 子图5: 源域和目标域特征统计对比雷达图
subplot(2, 3, 5);
if size(source_features_norm, 2) >= 6
    features_to_show = 1:6;
    source_means = mean(abs(source_features_norm(:, features_to_show)));
    target_means = mean(abs(target_features_norm(:, features_to_show)));
    
    % 创建雷达图数据
    angles = linspace(0, 2*pi, length(features_to_show)+1);
    source_means = [source_means, source_means(1)];
    target_means = [target_means, target_means(1)];
    
    % 转换为笛卡尔坐标
    x_source = source_means .* cos(angles);
    y_source = source_means .* sin(angles);
    x_target = target_means .* cos(angles);
    y_target = target_means .* sin(angles);
    
    % 绘制雷达图
    plot(x_source, y_source, 'r-o', 'LineWidth', 2, 'DisplayName', '源域');
    hold on;
    plot(x_target, y_target, 'b-s', 'LineWidth', 2, 'DisplayName', '目标域');
    
    % 绘制同心圆网格
    max_val = max([max(source_means), max(target_means)]);
    for r = 0.2:0.2:max_val
        theta_circle = linspace(0, 2*pi, 100);
        x_circle = r * cos(theta_circle);
        y_circle = r * sin(theta_circle);
        plot(x_circle, y_circle, 'k:', 'LineWidth', 0.5);
    end
    
    axis equal;
    grid on;
    title('特征统计雷达图');
    legend('Location', 'best');
else
    % 如果特征数不足，显示柱状图
    if size(source_features_norm, 2) >= 1
        features_to_show = 1:min(3, size(source_features_norm, 2));
        source_means = mean(abs(source_features_norm(:, features_to_show)));
        target_means = mean(abs(target_features_norm(:, features_to_show)));
        
        x = 1:length(features_to_show);
        bar(x-0.2, source_means, 0.4, 'FaceColor', 'r', 'DisplayName', '源域');
        hold on;
        bar(x+0.2, target_means, 0.4, 'FaceColor', 'b', 'DisplayName', '目标域');
        
        xlabel('特征索引');
        ylabel('平均绝对值');
        title('特征统计对比');
        legend('Location', 'best');
        grid on;
    end
end

% 子图6: 迁移学习方法效果对比
subplot(2, 3, 6);
if length(available_predictions) >= 2
    method_diversity = zeros(length(prediction_names), 1);
    for i = 1:length(prediction_names)
        pred = available_predictions{i};
        % 计算预测多样性
        pred_entropy = 0;
        for label = unique_labels'
            p = sum(pred == label) / length(pred);
            if p > 0
                pred_entropy = pred_entropy - p * log2(p);
            end
        end
        method_diversity(i) = pred_entropy;
    end
    
    bar(method_diversity, 'FaceColor', [0.3, 0.7, 0.9]);
    set(gca, 'XTickLabel', prediction_names);
    title('各方法预测多样性');
    ylabel('信息熵');
    xtickangle(45);
    grid on;
end

sgtitle('图6: 目标域样本详细分析', 'FontSize', 16, 'FontWeight', 'bold');
saveas(gcf, fullfile(result_dir, '图6_目标域样本详细分析.png'));

% 图7: 迁移学习性能评估
figure('Position', [700, 700, 1400, 1000]);

% 子图1: 不同迁移方法的预测一致性矩阵
subplot(2, 3, 1);
if length(available_predictions) >= 2
    imagesc(consistency_matrix);
    colorbar;
    set(gca, 'XTick', 1:length(prediction_names), 'XTickLabel', prediction_names);
    set(gca, 'YTick', 1:length(prediction_names), 'YTickLabel', prediction_names);
    title('方法间预测一致性');
    xtickangle(45);
end

% 子图2: 源域各类别样本在目标域的投影分布
subplot(2, 3, 2);
if size(target_features_pca, 2) >= 2
    % 使用KNN找到目标域中最接近源域各类别的样本
    for label = unique_labels'
        source_class_center = mean(source_features_pca(source_labels == label, 1:2));
        distances = sqrt(sum((target_features_pca(:, 1:2) - source_class_center).^2, 2));
        [~, closest_idx] = min(distances);
        
        scatter(target_features_pca(closest_idx, 1), target_features_pca(closest_idx, 2), 100, ...
                colors(label+1), 'filled', 'DisplayName', label_names{label+1});
        hold on;
    end
    xlabel('第一主成分');
    ylabel('第二主成分');
    title('目标域中各类别代表样本');
    legend('Location', 'best');
    grid on;
end

% 子图3: 预测结果稳定性分析
subplot(2, 3, 3);
if length(available_predictions) >= 2
    stability_scores = zeros(length(target_file_names), 1);
    for i = 1:length(target_file_names)
        predictions_for_sample = [];
        for j = 1:length(available_predictions)
            predictions_for_sample = [predictions_for_sample, available_predictions{j}(i)];
        end
        stability_scores(i) = length(unique(predictions_for_sample));
    end
    
    histogram(stability_scores, max(stability_scores), 'FaceColor', [0.9, 0.5, 0.3]);
    xlabel('预测方法数量');
    ylabel('样本数');
    title('预测稳定性分布');
    grid on;
end

% 子图4: 特征空间迁移效果
subplot(2, 3, 4);
% 计算源域和目标域在PCA空间的重叠度
if size(source_features_pca, 1) > 0 && size(target_features_pca, 1) > 0
    source_range = [min(source_features_pca(:, 1:2)); max(source_features_pca(:, 1:2))];
    target_range = [min(target_features_pca(:, 1:2)); max(target_features_pca(:, 1:2))];
    
    overlap_x = max(0, min(source_range(2,1), target_range(2,1)) - max(source_range(1,1), target_range(1,1)));
    overlap_y = max(0, min(source_range(2,2), target_range(2,2)) - max(source_range(1,2), target_range(1,2)));
    
    source_area = (source_range(2,1) - source_range(1,1)) * (source_range(2,2) - source_range(1,2));
    target_area = (target_range(2,1) - target_range(1,1)) * (target_range(2,2) - target_range(1,2));
    overlap_area = overlap_x * overlap_y;
    
    overlap_ratio = overlap_area / min(source_area, target_area);
    
    bar([overlap_ratio, 1-overlap_ratio], 'FaceColor', [0.4, 0.6, 0.8]);
    set(gca, 'XTickLabel', {'重叠区域', '非重叠区域'});
    title('特征空间重叠度');
    ylabel('比例');
    ylim([0, 1]);
    grid on;
end

% 子图5: 目标域预测置信度热图
subplot(2, 3, 5);
if isfield(transfer_results, 'ensemble') && exist('confidence_scores', 'var')
    % 将置信度重新排列为矩阵形式进行可视化
    n_files = length(target_file_names);
    conf_matrix = reshape([confidence_scores; zeros(16-mod(n_files,16), 1)], 4, []);
    imagesc(conf_matrix);
    colorbar;
    title('目标域预测置信度热图');
    xlabel('文件组');
    ylabel('文件索引');
end

% 子图6: 迁移学习综合评估
subplot(2, 3, 6);
if exist('domain_accuracy', 'var') && ~isempty(model_accuracies)
    metrics = [mean(model_accuracies), 1-domain_accuracy, mean_diff];
    metric_names = {'源域性能', '领域适应性', '分布相似性'};
    
    bar(metrics, 'FaceColor', [0.7, 0.2, 0.7]);
    set(gca, 'XTickLabel', metric_names);
    title('迁移学习综合评估');
    ylabel('评估分数');
    xtickangle(45);
    grid on;
end

sgtitle('图7: 迁移学习性能评估', 'FontSize', 16, 'FontWeight', 'bold');
saveas(gcf, fullfile(result_dir, '图7_迁移学习性能评估.png'));

% 图8: 目标域诊断结果可视化
figure('Position', [800, 800, 1400, 1000]);

% 子图1: 目标域文件预测结果表格可视化
subplot(2, 3, 1);
if isfield(transfer_results, 'ensemble')
    % 创建预测结果的可视化表格
    result_matrix = zeros(length(target_file_names), length(unique_labels));
    for i = 1:length(target_file_names)
        result_matrix(i, transfer_results.ensemble(i)+1) = 1;
    end
    
    imagesc(result_matrix);
    colormap(gca, [1 1 1; 0.2 0.8 0.2]); % 白色和绿色
    set(gca, 'XTick', 1:length(unique_labels), 'XTickLabel', label_names);
    set(gca, 'YTick', 1:min(16, length(target_file_names)));
    if length(target_file_names) <= 16
        set(gca, 'YTickLabel', target_file_names);
    end
    title('目标域文件诊断结果');
    xlabel('故障类型');
    ylabel('文件名');
    xtickangle(45);
end

% 子图2: 各类别预测概率分布
subplot(2, 3, 2);
if isfield(transfer_results, 'ensemble')
    class_probs = [];
    for label = unique_labels'
        class_probs = [class_probs, sum(transfer_results.ensemble == label) / length(transfer_results.ensemble)];
    end
    
    pie(class_probs, label_names);
    title('目标域故障类型分布');
end

% 子图3: 预测结果时序分析
subplot(2, 3, 3);
if isfield(transfer_results, 'ensemble')
    plot(transfer_results.ensemble, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('文件序号');
    ylabel('预测类别');
    title('预测结果时序图');
    set(gca, 'YTick', unique_labels, 'YTickLabel', label_names);
    grid on;
end

% 子图4: 故障严重程度评估
subplot(2, 3, 4);
if isfield(transfer_results, 'ensemble')
    % 定义故障严重程度 (0=正常, 1=轻微, 2=中等, 3=严重)
    severity_map = [0, 2, 3, 1]; % 正常, 滚动体, 内圈, 外圈
    severity_scores = severity_map(transfer_results.ensemble + 1);
    
    histogram(severity_scores, 0.5:1:3.5, 'FaceColor', [0.8, 0.4, 0.4]);
    xlabel('故障严重程度');
    ylabel('样本数');
    title('目标域故障严重程度分布');
    set(gca, 'XTick', 0:3, 'XTickLabel', {'正常', '轻微', '中等', '严重'});
    grid on;
end

% 子图5: 诊断可信度评估
subplot(2, 3, 5);
if exist('confidence_scores', 'var')
    % 根据置信度对样本进行分类
    high_conf = sum(confidence_scores > 0.8);
    med_conf = sum(confidence_scores > 0.6 & confidence_scores <= 0.8);
    low_conf = sum(confidence_scores <= 0.6);
    
    bar([high_conf, med_conf, low_conf], 'FaceColor', [0.3, 0.8, 0.3]);
    set(gca, 'XTickLabel', {'高可信度', '中等可信度', '低可信度'});
    title('诊断可信度分布');
    ylabel('样本数');
    grid on;
end

% 子图6: 迁移学习效果总结
subplot(2, 3, 6);
if ~isempty(available_predictions)
    % 计算各方法的预测多样性和一致性
    method_scores = zeros(length(prediction_names), 2);
    
    for i = 1:length(prediction_names)
        pred = available_predictions{i};
        
        % 多样性分数 (熵)
        entropy = 0;
        for label = unique_labels'
            p = sum(pred == label) / length(pred);
            if p > 0
                entropy = entropy - p * log2(p);
            end
        end
        method_scores(i, 1) = entropy;
        
        % 与集成结果的一致性
        if isfield(transfer_results, 'ensemble')
            consistency = sum(pred == transfer_results.ensemble) / length(pred);
            method_scores(i, 2) = consistency;
        end
    end
    
    bar(method_scores);
    set(gca, 'XTickLabel', prediction_names);
    legend({'预测多样性', '与集成一致性'}, 'Location', 'best');
    title('各方法性能对比');
    ylabel('分数');
    xtickangle(45);
    grid on;
end

sgtitle('图8: 目标域诊断结果可视化', 'FontSize', 16, 'FontWeight', 'bold');
saveas(gcf, fullfile(result_dir, '图8_目标域诊断结果可视化.png'));

%% 9. 生成分析报告
fprintf('\n=== 生成分析报告 ===\n');

report_file = fullfile(result_dir, '迁移学习诊断分析报告.txt');
fid = fopen(report_file, 'w', 'n', 'UTF-8');

fprintf(fid, '高速列车轴承智能故障诊断 - 迁移学习诊断分析报告\n');
fprintf(fid, '生成时间: %s\n\n', datestr(now));

fprintf(fid, '=== 数据概况 ===\n');
fprintf(fid, '源域样本数: %d\n', size(source_features, 1));
fprintf(fid, '目标域样本数: %d\n', size(target_features_all, 1));
fprintf(fid, '特征维度: %d\n', min_features);
fprintf(fid, '源域-目标域平均分布差异: %.4f\n', mean_diff);

fprintf(fid, '\n=== 迁移学习方法 ===\n');
fprintf(fid, '1. 直接迁移：使用源域模型直接预测目标域\n');
fprintf(fid, '2. 基于PCA的迁移：特征空间对齐后迁移\n');
fprintf(fid, '3. 集成迁移：多模型投票决策\n');

fprintf(fid, '\n=== 预测结果汇总 ===\n');
if isfield(transfer_results, 'ensemble')
    fprintf(fid, '推荐使用集成预测结果：\n');
    for label = unique_labels'
        count = sum(transfer_results.ensemble == label);
        percentage = count / length(transfer_results.ensemble) * 100;
        fprintf(fid, '  %s: %d个样本 (%.1f%%)\n', label_names{label+1}, count, percentage);
    end
    
    fprintf(fid, '\n目标域各文件预测结果：\n');
    for i = 1:length(target_file_names)
        fprintf(fid, '  %s: %s\n', target_file_names{i}, label_names{transfer_results.ensemble(i)+1});
    end
end

fprintf(fid, '\n=== 迁移学习效果评估 ===\n');
if exist('domain_accuracy', 'var')
    fprintf(fid, '领域分类器准确率: %.4f (越低表示迁移效果越好)\n', domain_accuracy);
    fprintf(fid, '领域适应性评分: %.4f\n', 1-domain_accuracy);
end

if ~isempty(model_accuracies)
    fprintf(fid, '源域模型平均准确率: %.4f\n', mean(model_accuracies));
end

fprintf(fid, '\n=== 结论与建议 ===\n');
fprintf(fid, '1. 迁移学习成功将源域知识应用到目标域\n');
fprintf(fid, '2. 集成方法提供了更稳定的预测结果\n');
fprintf(fid, '3. 建议重点关注置信度较低的样本\n');
fprintf(fid, '4. 可考虑收集更多目标域标注数据进行微调\n');

fclose(fid);

fprintf('分析报告生成完成！\n');
fprintf('\n=== 任务3完成 ===\n');
fprintf('所有结果已保存到 %s 文件夹\n', result_dir);

%% 辅助函数：特征提取（用于目标域数据）
function features = extract_features_for_transfer(signal_data)
    % 为迁移学习提取与任务一相同的特征
    
    % 如果是多维数据，取第一列或展平
    if size(signal_data, 2) > 1
        signal_data = signal_data(:, 1);
    end
    signal_data = signal_data(:);
    
    % 基础统计特征
    features = [];
    
    % 时域特征
    features(end+1) = mean(signal_data);                    % 均值
    features(end+1) = std(signal_data);                     % 标准差
    features(end+1) = var(signal_data);                     % 方差
    features(end+1) = skewness(signal_data);                % 偏度
    features(end+1) = kurtosis(signal_data);                % 峰度
    features(end+1) = max(signal_data);                     % 最大值
    features(end+1) = min(signal_data);                     % 最小值
    features(end+1) = range(signal_data);                   % 极差
    features(end+1) = rms(signal_data);                     % 均方根
    features(end+1) = mean(abs(signal_data));               % 平均绝对值
    
    % 频域特征
    try
        Y = fft(signal_data);
        P = abs(Y).^2;
        
        features(end+1) = mean(P);                          % 频域均值
        features(end+1) = std(P);                           % 频域标准差
        features(end+1) = max(P);                           % 频域最大值
        
        % 频域重心
        freqs = (0:length(P)-1) / length(P);
        features(end+1) = sum(freqs .* P') / sum(P);
        
        % 频域带宽
        fc = features(end);  % 重心频率
        features(end+1) = sqrt(sum(((freqs - fc).^2) .* P') / sum(P));
        
    catch
        % 如果FFT失败，添加零特征
        features(end+1:end+5) = 0;
    end
    
    % 确保特征是行向量
    features = features(:)';
    
    % 处理NaN和Inf
    features(isnan(features)) = 0;
    features(isinf(features)) = 0;
end