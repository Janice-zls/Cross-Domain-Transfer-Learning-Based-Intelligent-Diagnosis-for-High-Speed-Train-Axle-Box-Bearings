%% 高速列车轴承智能故障诊断 - 任务2：源域故障诊断

clear; clc; close all;

%% 1. 加载任务一的特征数据
fprintf('=== 任务2：源域故障诊断 ===\n');
fprintf('加载任务一提取的特征数据...\n');

% 加载任务一的结果
task1_data_path = 'c:\Users\Jack\Desktop\E题\问题一结果\task1_complete_data.mat';
if ~exist(task1_data_path, 'file')
    error('未找到任务一的数据文件，请先运行任务一！');
end

load(task1_data_path);

% 创建结果保存文件夹
result_dir = '问题二结果';
if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

fprintf('数据加载完成！\n');
fprintf('总样本数: %d\n', size(feature_matrix, 1));
fprintf('特征维度: %d\n', size(feature_matrix, 2));

% 显示类别分布
unique_labels = unique(labels);
label_names = {'正常', '滚动体故障', '内圈故障', '外圈故障'};
fprintf('\n类别分布:\n');
for i = 1:length(unique_labels)
    label = unique_labels(i);
    count = sum(labels == label);
    percentage = count / length(labels) * 100;
    fprintf('%s: %d 个样本 (%.1f%%)\n', label_names{label+1}, count, percentage);
end

%% 2. 数据预处理和划分
fprintf('\n=== 数据预处理和划分 ===\n');

% 2.1 特征预处理
fprintf('进行特征预处理...\n');

% 移除常数特征（方差为0的特征）
feature_vars = var(feature_matrix);
constant_features = find(feature_vars == 0);
if ~isempty(constant_features)
    fprintf('发现 %d 个常数特征，将其移除\n', length(constant_features));
    feature_matrix(:, constant_features) = [];
    fprintf('移除常数特征后，特征维度: %d\n', size(feature_matrix, 2));
end

% 移除包含NaN或Inf的特征
nan_features = any(isnan(feature_matrix) | isinf(feature_matrix), 1);
if sum(nan_features) > 0
    fprintf('发现 %d 个包含NaN/Inf的特征，将其移除\n', sum(nan_features));
    feature_matrix(:, nan_features) = [];
    fprintf('移除异常特征后，特征维度: %d\n', size(feature_matrix, 2));
end

% 特征标准化
feature_matrix_normalized = zscore(feature_matrix);

% 处理标准化后的NaN值
nan_indices = any(isnan(feature_matrix_normalized), 2);
if sum(nan_indices) > 0
    fprintf('发现 %d 个包含NaN的样本，将其移除\n', sum(nan_indices));
    feature_matrix_normalized = feature_matrix_normalized(~nan_indices, :);
    labels = labels(~nan_indices);
end

% 添加小的随机噪声以避免方差为0的问题
noise_level = 1e-6;
feature_matrix_normalized = feature_matrix_normalized + noise_level * randn(size(feature_matrix_normalized));

fprintf('特征预处理完成！最终特征维度: %d\n', size(feature_matrix_normalized, 2));

% 2.2 数据集划分（分层抽样确保各类别比例一致）
fprintf('划分训练集和测试集...\n');
train_ratio = 0.7;
test_ratio = 0.3;

train_indices = [];
test_indices = [];

% 对每个类别进行分层抽样
for label = unique_labels'
    class_indices = find(labels == label);
    n_class = length(class_indices);
    n_train = round(n_class * train_ratio);
    
    % 随机打乱索引
    shuffled_indices = class_indices(randperm(n_class));
    
    train_indices = [train_indices; shuffled_indices(1:n_train)];
    test_indices = [test_indices; shuffled_indices(n_train+1:end)];
end

% 提取训练集和测试集
X_train = feature_matrix_normalized(train_indices, :);
y_train = labels(train_indices);
X_test = feature_matrix_normalized(test_indices, :);
y_test = labels(test_indices);

fprintf('训练集样本数: %d\n', size(X_train, 1));
fprintf('测试集样本数: %d\n', size(X_test, 1));

% 显示训练集和测试集的类别分布
fprintf('\n训练集类别分布:\n');
train_class_counts = [];
for i = 1:length(unique_labels)
    label = unique_labels(i);
    count = sum(y_train == label);
    percentage = count / length(y_train) * 100;
    fprintf('%s: %d 个样本 (%.1f%%)\n', label_names{label+1}, count, percentage);
    train_class_counts = [train_class_counts; count];
end

fprintf('\n测试集类别分布:\n');
test_class_counts = [];
for i = 1:length(unique_labels)
    label = unique_labels(i);
    count = sum(y_test == label);
    percentage = count / length(y_test) * 100;
    fprintf('%s: %d 个样本 (%.1f%%)\n', label_names{label+1}, count, percentage);
    test_class_counts = [test_class_counts; count];
end

% 保存数据划分信息
data_split_info = table(label_names', train_class_counts, test_class_counts, ...
    'VariableNames', {'类别', '训练集样本数', '测试集样本数'});
writetable(data_split_info, fullfile(result_dir, '数据划分信息.csv'));

%% 3. 模型训练和评估
fprintf('\n=== 模型训练和评估 ===\n');

% 存储所有模型的结果
model_results = struct();

%% 3.1 支持向量机 (SVM)
fprintf('\n--- 训练SVM模型 ---\n');
tic;

try
    % 使用多分类SVM
    svm_template = templateSVM('KernelFunction', 'rbf', 'Standardize', false);
    svm_model = fitcecoc(X_train, y_train, 'Learners', svm_template);
    
    % 预测
    y_pred_svm = predict(svm_model, X_test);
    
    % 计算评估指标
    svm_accuracy = sum(y_pred_svm == y_test) / length(y_test);
    svm_time = toc;
    
    fprintf('SVM训练完成，用时: %.2f秒\n', svm_time);
    fprintf('SVM测试准确率: %.4f\n', svm_accuracy);
    
    % 计算混淆矩阵
    svm_cm = confusionmat(y_test, y_pred_svm);
    model_results.svm.accuracy = svm_accuracy;
    model_results.svm.confusion_matrix = svm_cm;
    model_results.svm.predictions = y_pred_svm;
    model_results.svm.training_time = svm_time;
    model_results.svm.success = true;
    
catch ME
    fprintf('SVM训练失败: %s\n', ME.message);
    model_results.svm.success = false;
    model_results.svm.accuracy = 0;
    model_results.svm.training_time = toc;
end

%% 3.2 随机森林 (Random Forest)
fprintf('\n--- 训练随机森林模型 ---\n');
tic;

try
    % 训练随机森林
    rf_model = TreeBagger(100, X_train, y_train, 'Method', 'classification', ...
                          'NumPredictorsToSample', 'all', 'OOBPrediction', 'on');
    
    % 预测
    y_pred_rf_cell = predict(rf_model, X_test);
    y_pred_rf = str2double(y_pred_rf_cell);
    
    % 计算评估指标
    rf_accuracy = sum(y_pred_rf == y_test) / length(y_test);
    rf_time = toc;
    
    fprintf('随机森林训练完成，用时: %.2f秒\n', rf_time);
    fprintf('随机森林测试准确率: %.4f\n', rf_accuracy);
    
    % 计算混淆矩阵
    rf_cm = confusionmat(y_test, y_pred_rf);
    model_results.rf.accuracy = rf_accuracy;
    model_results.rf.confusion_matrix = rf_cm;
    model_results.rf.predictions = y_pred_rf;
    model_results.rf.training_time = rf_time;
    model_results.rf.success = true;
    
catch ME
    fprintf('随机森林训练失败: %s\n', ME.message);
    model_results.rf.success = false;
    model_results.rf.accuracy = 0;
    model_results.rf.training_time = toc;
end

%% 3.3 K近邻 (KNN)
fprintf('\n--- 训练KNN模型 ---\n');
tic;

try
    % 训练KNN模型
    knn_model = fitcknn(X_train, y_train, 'NumNeighbors', 5, 'Distance', 'euclidean');
    
    % 预测
    y_pred_knn = predict(knn_model, X_test);
    
    % 计算评估指标
    knn_accuracy = sum(y_pred_knn == y_test) / length(y_test);
    knn_time = toc;
    
    fprintf('KNN训练完成，用时: %.2f秒\n', knn_time);
    fprintf('KNN测试准确率: %.4f\n', knn_accuracy);
    
    % 计算混淆矩阵
    knn_cm = confusionmat(y_test, y_pred_knn);
    model_results.knn.accuracy = knn_accuracy;
    model_results.knn.confusion_matrix = knn_cm;
    model_results.knn.predictions = y_pred_knn;
    model_results.knn.training_time = knn_time;
    model_results.knn.success = true;
    
catch ME
    fprintf('KNN训练失败: %s\n', ME.message);
    model_results.knn.success = false;
    model_results.knn.accuracy = 0;
    model_results.knn.training_time = toc;
end

%% 3.4 朴素贝叶斯 (Naive Bayes) - 改进版
fprintf('\n--- 训练朴素贝叶斯模型 ---\n');
tic;

try
    % 使用改进的朴素贝叶斯训练方法
    % 设置分布类型为kernel以避免方差为0的问题
    nb_model = fitcnb(X_train, y_train, 'DistributionNames', 'kernel');
    
    % 预测
    y_pred_nb = predict(nb_model, X_test);
    
    % 计算评估指标
    nb_accuracy = sum(y_pred_nb == y_test) / length(y_test);
    nb_time = toc;
    
    fprintf('朴素贝叶斯训练完成，用时: %.2f秒\n', nb_time);
    fprintf('朴素贝叶斯测试准确率: %.4f\n', nb_accuracy);
    
    % 计算混淆矩阵
    nb_cm = confusionmat(y_test, y_pred_nb);
    model_results.nb.accuracy = nb_accuracy;
    model_results.nb.confusion_matrix = nb_cm;
    model_results.nb.predictions = y_pred_nb;
    model_results.nb.training_time = nb_time;
    model_results.nb.success = true;
    
catch ME
    fprintf('朴素贝叶斯训练失败，尝试使用多项式分布: %s\n', ME.message);
    try
        % 如果kernel失败，尝试使用mn（多项式）分布
        % 首先将数据转换为非负值
        X_train_pos = X_train - min(X_train(:)) + 1;
        X_test_pos = X_test - min(X_test(:)) + 1;
        
        nb_model = fitcnb(X_train_pos, y_train, 'DistributionNames', 'mn');
        y_pred_nb = predict(nb_model, X_test_pos);
        
        nb_accuracy = sum(y_pred_nb == y_test) / length(y_test);
        nb_time = toc;
        
        fprintf('朴素贝叶斯(多项式)训练完成，用时: %.2f秒\n', nb_time);
        fprintf('朴素贝叶斯测试准确率: %.4f\n', nb_accuracy);
        
        nb_cm = confusionmat(y_test, y_pred_nb);
        model_results.nb.accuracy = nb_accuracy;
        model_results.nb.confusion_matrix = nb_cm;
        model_results.nb.predictions = y_pred_nb;
        model_results.nb.training_time = nb_time;
        model_results.nb.success = true;
        
    catch ME2
        fprintf('朴素贝叶斯训练完全失败: %s\n', ME2.message);
        model_results.nb.success = false;
        model_results.nb.accuracy = 0;
        model_results.nb.training_time = toc;
    end
end

%% 3.5 决策树 (Decision Tree)
fprintf('\n--- 训练决策树模型 ---\n');
tic;

try
    % 训练决策树模型
    dt_model = fitctree(X_train, y_train);
    
    % 预测
    y_pred_dt = predict(dt_model, X_test);
    
    % 计算评估指标
    dt_accuracy = sum(y_pred_dt == y_test) / length(y_test);
    dt_time = toc;
    
    fprintf('决策树训练完成，用时: %.2f秒\n', dt_time);
    fprintf('决策树测试准确率: %.4f\n', dt_accuracy);
    
    % 计算混淆矩阵
    dt_cm = confusionmat(y_test, y_pred_dt);
    model_results.dt.accuracy = dt_accuracy;
    model_results.dt.confusion_matrix = dt_cm;
    model_results.dt.predictions = y_pred_dt;
    model_results.dt.training_time = dt_time;
    model_results.dt.success = true;
    
catch ME
    fprintf('决策树训练失败: %s\n', ME.message);
    model_results.dt.success = false;
    model_results.dt.accuracy = 0;
    model_results.dt.training_time = toc;
end

%% 4. 详细评估指标计算
fprintf('\n=== 详细评估指标计算 ===\n');

models = {'svm', 'rf', 'knn', 'nb', 'dt'};
model_names = {'SVM', '随机森林', 'KNN', '朴素贝叶斯', '决策树'};
successful_models = {};
successful_model_names = {};

% 只处理成功训练的模型
for i = 1:length(models)
    model_name = models{i};
    if model_results.(model_name).success
        successful_models{end+1} = model_name;
        successful_model_names{end+1} = model_names{i};
    end
end

fprintf('成功训练的模型数量: %d\n', length(successful_models));

% 计算每个成功模型的详细指标
detailed_metrics = [];
for i = 1:length(successful_models)
    model_name = successful_models{i};
    display_name = successful_model_names{i};
    
    cm = model_results.(model_name).confusion_matrix;
    
    % 计算精确率、召回率、F1分数
    n_classes = length(unique_labels);
    precision = zeros(n_classes, 1);
    recall = zeros(n_classes, 1);
    f1_score = zeros(n_classes, 1);
    
    for j = 1:n_classes
        tp = cm(j, j);
        fp = sum(cm(:, j)) - tp;
        fn = sum(cm(j, :)) - tp;
        
        precision(j) = tp / (tp + fp);
        recall(j) = tp / (tp + fn);
        f1_score(j) = 2 * precision(j) * recall(j) / (precision(j) + recall(j));
        
        % 处理除零情况
        if isnan(precision(j)), precision(j) = 0; end
        if isnan(recall(j)), recall(j) = 0; end
        if isnan(f1_score(j)), f1_score(j) = 0; end
    end
    
    % 计算宏平均
    macro_precision = mean(precision);
    macro_recall = mean(recall);
    macro_f1 = mean(f1_score);
    
    % 存储结果
    model_results.(model_name).precision = precision;
    model_results.(model_name).recall = recall;
    model_results.(model_name).f1_score = f1_score;
    model_results.(model_name).macro_precision = macro_precision;
    model_results.(model_name).macro_recall = macro_recall;
    model_results.(model_name).macro_f1 = macro_f1;
    
    % 收集详细指标用于保存
    detailed_metrics = [detailed_metrics; model_results.(model_name).accuracy, ...
        macro_precision, macro_recall, macro_f1, model_results.(model_name).training_time];
    
    fprintf('\n%s 详细评估结果:\n', display_name);
    fprintf('准确率: %.4f\n', model_results.(model_name).accuracy);
    fprintf('宏平均精确率: %.4f\n', macro_precision);
    fprintf('宏平均召回率: %.4f\n', macro_recall);
    fprintf('宏平均F1分数: %.4f\n', macro_f1);
    
    fprintf('各类别详细指标:\n');
    for j = 1:n_classes
        fprintf('  %s - 精确率: %.4f, 召回率: %.4f, F1: %.4f\n', ...
            label_names{j}, precision(j), recall(j), f1_score(j));
    end
    
    % 保存每个模型的详细指标
    model_detail_table = table(label_names', precision, recall, f1_score, ...
        'VariableNames', {'类别', '精确率', '召回率', 'F1分数'});
    writetable(model_detail_table, fullfile(result_dir, sprintf('%s_详细指标.csv', display_name)));
    
    % 保存混淆矩阵
    cm_table = array2table(cm, 'VariableNames', label_names, 'RowNames', label_names);
    writetable(cm_table, fullfile(result_dir, sprintf('%s_混淆矩阵.csv', display_name)), 'WriteRowNames', true);
end

% 保存所有成功模型的综合指标
if ~isempty(detailed_metrics)
    comprehensive_metrics = table(successful_model_names', detailed_metrics(:,1), detailed_metrics(:,2), ...
        detailed_metrics(:,3), detailed_metrics(:,4), detailed_metrics(:,5), ...
        'VariableNames', {'模型', '准确率', '宏平均精确率', '宏平均召回率', '宏平均F1分数', '训练时间'});
    writetable(comprehensive_metrics, fullfile(result_dir, '所有模型综合指标.csv'));
end

%% 5. 交叉验证（仅对成功的模型）
fprintf('\n=== 5折交叉验证 ===\n');

if ~isempty(successful_models)
    k_folds = 5;
    cv_results = struct();
    cv_detailed_results = [];
    
    for i = 1:length(successful_models)
        model_name = successful_models{i};
        display_name = successful_model_names{i};
        
        fprintf('进行%s的5折交叉验证...\n', display_name);
        
        % 创建交叉验证分区
        cv_partition = cvpartition(y_train, 'KFold', k_folds);
        cv_accuracies = zeros(k_folds, 1);
        
        for fold = 1:k_folds
            train_idx = training(cv_partition, fold);
            val_idx = test(cv_partition, fold);
            
            X_cv_train = X_train(train_idx, :);
            y_cv_train = y_train(train_idx);
            X_cv_val = X_train(val_idx, :);
            y_cv_val = y_train(val_idx);
            
            try
                % 根据模型类型训练
                switch model_name
                    case 'svm'
                        cv_model = fitcecoc(X_cv_train, y_cv_train, 'Learners', svm_template);
                        y_cv_pred = predict(cv_model, X_cv_val);
                    case 'rf'
                        cv_model = TreeBagger(50, X_cv_train, y_cv_train, 'Method', 'classification');
                        y_cv_pred_cell = predict(cv_model, X_cv_val);
                        y_cv_pred = str2double(y_cv_pred_cell);
                    case 'knn'
                        cv_model = fitcknn(X_cv_train, y_cv_train, 'NumNeighbors', 5);
                        y_cv_pred = predict(cv_model, X_cv_val);
                    case 'nb'
                        % 使用kernel分布的朴素贝叶斯
                        cv_model = fitcnb(X_cv_train, y_cv_train, 'DistributionNames', 'kernel');
                        y_cv_pred = predict(cv_model, X_cv_val);
                    case 'dt'
                        cv_model = fitctree(X_cv_train, y_cv_train);
                        y_cv_pred = predict(cv_model, X_cv_val);
                end
                
                cv_accuracies(fold) = sum(y_cv_pred == y_cv_val) / length(y_cv_val);
                
            catch ME
                fprintf('第%d折交叉验证失败: %s\n', fold, ME.message);
                cv_accuracies(fold) = 0;
            end
        end
        
        cv_results.(model_name).mean_accuracy = mean(cv_accuracies);
        cv_results.(model_name).std_accuracy = std(cv_accuracies);
        cv_results.(model_name).all_accuracies = cv_accuracies;
        
        cv_detailed_results = [cv_detailed_results; cv_accuracies'];
        
        fprintf('%s 交叉验证结果: %.4f ± %.4f\n', display_name, ...
            cv_results.(model_name).mean_accuracy, cv_results.(model_name).std_accuracy);
    end
    
    % 保存交叉验证结果
    if ~isempty(cv_detailed_results)
        cv_table = array2table(cv_detailed_results', 'VariableNames', successful_model_names);
        writetable(cv_table, fullfile(result_dir, '交叉验证详细结果.csv'));
        
        cv_means = [];
        cv_stds = [];
        for i = 1:length(successful_models)
            model_name = successful_models{i};
            cv_means = [cv_means; cv_results.(model_name).mean_accuracy];
            cv_stds = [cv_stds; cv_results.(model_name).std_accuracy];
        end
        
        cv_summary = table(successful_model_names', cv_means, cv_stds, ...
            'VariableNames', {'模型', '平均准确率', '标准差'});
        writetable(cv_summary, fullfile(result_dir, '交叉验证汇总.csv'));
    end
end

%% 6. 结果可视化（仅对成功的模型）
fprintf('\n=== 生成结果可视化 ===\n');

if ~isempty(successful_models)
    % 提取成功模型的数据
    success_accuracies = [];
    success_training_times = [];
    success_cv_means = [];
    success_cv_stds = [];
    
    for i = 1:length(successful_models)
        model_name = successful_models{i};
        success_accuracies = [success_accuracies, model_results.(model_name).accuracy];
        success_training_times = [success_training_times, model_results.(model_name).training_time];
        if isfield(cv_results, model_name)
            success_cv_means = [success_cv_means, cv_results.(model_name).mean_accuracy];
            success_cv_stds = [success_cv_stds, cv_results.(model_name).std_accuracy];
        end
    end
    
    % 图1: 基础性能对比
    figure('Position', [100, 100, 1200, 800]);
    
    % 子图1: 模型准确率对比
    subplot(2, 2, 1);
    bar(success_accuracies, 'FaceColor', [0.2 0.6 0.8]);
    set(gca, 'XTickLabel', successful_model_names);
    title('模型准确率对比');
    ylabel('准确率');
    ylim([0, 1]);
    grid on;
    xtickangle(45);
    
    % 添加数值标签
    for i = 1:length(success_accuracies)
        text(i, success_accuracies(i) + 0.02, sprintf('%.3f', success_accuracies(i)), ...
             'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    end
    
    % 子图2: 训练时间对比
    subplot(2, 2, 2);
    bar(success_training_times, 'FaceColor', [0.8 0.4 0.2]);
    set(gca, 'XTickLabel', successful_model_names);
    title('训练时间对比');
    ylabel('时间 (秒)');
    grid on;
    xtickangle(45);
    
    % 子图3: 交叉验证结果
    if ~isempty(success_cv_means)
        subplot(2, 2, 3);
        bar(success_cv_means, 'FaceColor', [0.4 0.8 0.4]);
        hold on;
        errorbar(1:length(success_cv_means), success_cv_means, success_cv_stds, 'k.', 'LineWidth', 2);
        set(gca, 'XTickLabel', successful_model_names);
        title('交叉验证准确率');
        ylabel('准确率');
        ylim([0, 1]);
        grid on;
        xtickangle(45);
    end
    
    % 子图4: 综合性能对比
    subplot(2, 2, 4);
    if ~isempty(detailed_metrics)
        bar(detailed_metrics(:, 1:4)');
        legend(successful_model_names, 'Location', 'best');
        set(gca, 'XTickLabel', {'准确率', '精确率', '召回率', 'F1分数'});
        title('模型综合性能对比');
        ylabel('分数');
        grid on;
        xtickangle(45);
    end
    
    sgtitle('图1: 基础性能对比', 'FontSize', 16, 'FontWeight', 'bold');
    saveas(gcf, fullfile(result_dir, '图1_基础性能对比.png'));
    
    % 图2: 混淆矩阵集合
    if length(successful_models) > 0
        figure('Position', [200, 200, 1600, 1000]);
        n_models = length(successful_models);
        n_cols = ceil(sqrt(n_models));
        n_rows = ceil(n_models / n_cols);
        
        for i = 1:n_models
            subplot(n_rows, n_cols, i);
            model_name = successful_models{i};
            display_name = successful_model_names{i};
            
            cm = model_results.(model_name).confusion_matrix;
            
            % 绘制混淆矩阵热图
            imagesc(cm);
            colormap(hot);
            colorbar;
            
            % 添加数值标签
            for row = 1:size(cm, 1)
                for col = 1:size(cm, 2)
                    text(col, row, num2str(cm(row, col)), ...
                         'HorizontalAlignment', 'center', 'Color', 'white', 'FontWeight', 'bold');
                end
            end
            
            set(gca, 'XTick', 1:length(label_names), 'XTickLabel', label_names);
            set(gca, 'YTick', 1:length(label_names), 'YTickLabel', label_names);
            xlabel('预测类别');
            ylabel('真实类别');
            title(sprintf('%s混淆矩阵', display_name));
            xtickangle(45);
        end
        
        sgtitle('图2: 混淆矩阵对比', 'FontSize', 16, 'FontWeight', 'bold');
        saveas(gcf, fullfile(result_dir, '图2_混淆矩阵对比.png'));
    end
    
    % 图3: 类别分布可视化
    figure('Position', [300, 300, 1200, 800]);
    
    % 子图1: 原始数据类别分布
    subplot(2, 2, 1);
    class_counts = [];
    for i = 1:length(unique_labels)
        class_counts = [class_counts, sum(labels == unique_labels(i))];
    end
    pie(class_counts, label_names);
    title('原始数据类别分布');
    
    % 子图2: 训练集类别分布
    subplot(2, 2, 2);
    train_counts = [];
    for i = 1:length(unique_labels)
        train_counts = [train_counts, sum(y_train == unique_labels(i))];
    end
    bar(train_counts, 'FaceColor', [0.3 0.7 0.9]);
    set(gca, 'XTickLabel', label_names);
    title('训练集类别分布');
    ylabel('样本数');
    xtickangle(45);
    
    % 子图3: 测试集类别分布
    subplot(2, 2, 3);
    test_counts = [];
    for i = 1:length(unique_labels)
        test_counts = [test_counts, sum(y_test == unique_labels(i))];
    end
    bar(test_counts, 'FaceColor', [0.9 0.7 0.3]);
    set(gca, 'XTickLabel', label_names);
    title('测试集类别分布');
    ylabel('样本数');
    xtickangle(45);
    
    % 子图4: 训练测试集比例对比
    subplot(2, 2, 4);
    ratio_data = [train_counts; test_counts]';
    bar(ratio_data, 'grouped');
    set(gca, 'XTickLabel', label_names);
    title('训练测试集比例对比');
    ylabel('样本数');
    legend({'训练集', '测试集'}, 'Location', 'best');
    xtickangle(45);
    
    sgtitle('图3: 数据分布分析', 'FontSize', 16, 'FontWeight', 'bold');
    saveas(gcf, fullfile(result_dir, '图3_数据分布分析.png'));
    
    % 图4: 特征分布箱线图
    figure('Position', [400, 400, 1400, 900]);
    
    % 选择前12个最重要的特征进行可视化
    n_features_to_show = min(12, size(X_train, 2));
    feature_indices = 1:n_features_to_show;
    
    for i = 1:n_features_to_show
        subplot(3, 4, i);
        
        % 为每个类别创建箱线图数据
        boxplot_data = [];
        group_labels = [];
        
        for j = 1:length(unique_labels)
            class_data = X_train(y_train == unique_labels(j), feature_indices(i));
            boxplot_data = [boxplot_data; class_data];
            group_labels = [group_labels; repmat(j, length(class_data), 1)];
        end
        
        boxplot(boxplot_data, group_labels);
        set(gca, 'XTickLabel', label_names);
        title(sprintf('特征%d分布', feature_indices(i)));
        ylabel('特征值');
        xtickangle(45);
    end
    
    sgtitle('图4: 特征分布箱线图', 'FontSize', 16, 'FontWeight', 'bold');
    saveas(gcf, fullfile(result_dir, '图4_特征分布箱线图.png'));
    
    % 图5: 模型性能雷达图
    if length(successful_models) >= 2
        figure('Position', [500, 500, 1000, 800]);
        
        % 准备雷达图数据
        metrics_names = {'准确率', '精确率', '召回率', 'F1分数', '稳定性'};
        n_metrics = length(metrics_names);
        
        % 计算稳定性指标（1 - CV标准差）
        stability_scores = [];
        for i = 1:length(successful_models)
            model_name = successful_models{i};
            if isfield(cv_results, model_name)
                stability = 1 - cv_results.(model_name).std_accuracy;
            else
                stability = 0.8; % 默认值
            end
            stability_scores = [stability_scores, stability];
        end
        
        % 创建雷达图数据矩阵
        radar_data = [success_accuracies; detailed_metrics(:,2)'; detailed_metrics(:,3)'; 
                     detailed_metrics(:,4)'; stability_scores];
        
        % 绘制雷达图
        angles = linspace(0, 2*pi, n_metrics+1);
        colors = lines(length(successful_models));
        
        for i = 1:length(successful_models)
            model_data = [radar_data(:,i); radar_data(1,i)]; % 闭合图形
            polarplot(angles, model_data, 'o-', 'LineWidth', 2, 'Color', colors(i,:));
            hold on;
        end
        
        % 设置雷达图属性
        thetaticks(angles(1:end-1) * 180/pi);
        thetaticklabels(metrics_names);
        rlim([0, 1]);
        title('图5: 模型性能雷达图');
        legend(successful_model_names, 'Location', 'best');
        
        saveas(gcf, fullfile(result_dir, '图5_模型性能雷达图.png'));
    end
    
    % 图6: 学习曲线分析
    figure('Position', [600, 600, 1400, 1000]);
    
    % 为每个成功的模型绘制学习曲线
    n_models = length(successful_models);
    n_cols = ceil(sqrt(n_models));
    n_rows = ceil(n_models / n_cols);
    
    for i = 1:n_models
        subplot(n_rows, n_cols, i);
        model_name = successful_models{i};
        display_name = successful_model_names{i};
        
        % 生成不同训练集大小的学习曲线
        train_sizes = round(linspace(0.1, 1.0, 10) * size(X_train, 1));
        train_scores = [];
        val_scores = [];
        
        for size_idx = 1:length(train_sizes)
            train_size = train_sizes(size_idx);
            
            % 随机选择训练样本
            rand_indices = randperm(size(X_train, 1), train_size);
            X_subset = X_train(rand_indices, :);
            y_subset = y_train(rand_indices);
            
            try
                % 训练模型
                switch model_name
                    case 'svm'
                        temp_model = fitcecoc(X_subset, y_subset, 'Learners', svm_template);
                    case 'rf'
                        temp_model = TreeBagger(50, X_subset, y_subset, 'Method', 'classification');
                    case 'knn'
                        temp_model = fitcknn(X_subset, y_subset, 'NumNeighbors', 5);
                    case 'nb'
                        temp_model = fitcnb(X_subset, y_subset, 'DistributionNames', 'kernel');
                    case 'dt'
                        temp_model = fitctree(X_subset, y_subset);
                end
                
                % 计算训练和验证准确率
                if strcmp(model_name, 'rf')
                    train_pred_cell = predict(temp_model, X_subset);
                    train_pred = str2double(train_pred_cell);
                    val_pred_cell = predict(temp_model, X_test);
                    val_pred = str2double(val_pred_cell);
                else
                    train_pred = predict(temp_model, X_subset);
                    val_pred = predict(temp_model, X_test);
                end
                
                train_acc = sum(train_pred == y_subset) / length(y_subset);
                val_acc = sum(val_pred == y_test) / length(y_test);
                
                train_scores = [train_scores, train_acc];
                val_scores = [val_scores, val_acc];
                
            catch
                train_scores = [train_scores, 0];
                val_scores = [val_scores, 0];
            end
        end
        
        plot(train_sizes, train_scores, 'o-', 'LineWidth', 2, 'DisplayName', '训练准确率');
        hold on;
        plot(train_sizes, val_scores, 's-', 'LineWidth', 2, 'DisplayName', '验证准确率');
        
        xlabel('训练样本数');
        ylabel('准确率');
        title(sprintf('%s学习曲线', display_name));
        legend('Location', 'best');
        grid on;
    end
    
    sgtitle('图6: 学习曲线分析', 'FontSize', 16, 'FontWeight', 'bold');
    saveas(gcf, fullfile(result_dir, '图6_学习曲线分析.png'));
    
    % 图7: 详细性能指标对比
    figure('Position', [700, 700, 1400, 1000]);
    
    % 子图1: 各类别精确率对比
    subplot(2, 2, 1);
    precision_matrix = [];
    for i = 1:length(successful_models)
        model_name = successful_models{i};
        precision_matrix = [precision_matrix, model_results.(model_name).precision];
    end
    bar(precision_matrix');
    set(gca, 'XTickLabel', successful_model_names);
    title('各类别精确率对比');
    ylabel('精确率');
    legend(label_names, 'Location', 'best');
    xtickangle(45);
    
    % 子图2: 各类别召回率对比
    subplot(2, 2, 2);
    recall_matrix = [];
    for i = 1:length(successful_models)
        model_name = successful_models{i};
        recall_matrix = [recall_matrix, model_results.(model_name).recall];
    end
    bar(recall_matrix');
    set(gca, 'XTickLabel', successful_model_names);
    title('各类别召回率对比');
    ylabel('召回率');
    legend(label_names, 'Location', 'best');
    xtickangle(45);
    
    % 子图3: 各类别F1分数对比
    subplot(2, 2, 3);
    f1_matrix = [];
    for i = 1:length(successful_models)
        model_name = successful_models{i};
        f1_matrix = [f1_matrix, model_results.(model_name).f1_score];
    end
    bar(f1_matrix');
    set(gca, 'XTickLabel', successful_model_names);
    title('各类别F1分数对比');
    ylabel('F1分数');
    legend(label_names, 'Location', 'best');
    xtickangle(45);
    
    % 子图4: 模型复杂度vs性能
    subplot(2, 2, 4);
    % 定义模型复杂度（相对值）
    complexity_scores = [];
    for i = 1:length(successful_models)
        model_name = successful_models{i};
        switch model_name
            case 'knn'
                complexity = 0.2;
            case 'nb'
                complexity = 0.3;
            case 'dt'
                complexity = 0.5;
            case 'svm'
                complexity = 0.7;
            case 'rf'
                complexity = 0.8;
            otherwise
                complexity = 0.5;
        end
        complexity_scores = [complexity_scores, complexity];
    end
    
    scatter(complexity_scores, success_accuracies, 100, 'filled');
    for i = 1:length(successful_models)
        text(complexity_scores(i), success_accuracies(i) + 0.01, ...
             successful_model_names{i}, 'HorizontalAlignment', 'center');
    end
    xlabel('模型复杂度');
    ylabel('准确率');
    title('模型复杂度vs性能');
    grid on;
    
    sgtitle('图7: 详细性能指标对比', 'FontSize', 16, 'FontWeight', 'bold');
    saveas(gcf, fullfile(result_dir, '图7_详细性能指标对比.png'));
    
    % 图8: 交叉验证详细分析
    if ~isempty(success_cv_means)
        figure('Position', [800, 800, 1200, 800]);
        
        % 子图1: 交叉验证箱线图
        subplot(2, 2, 1);
        cv_data_matrix = [];
        for i = 1:length(successful_models)
            model_name = successful_models{i};
            if isfield(cv_results, model_name)
                cv_data_matrix = [cv_data_matrix, cv_results.(model_name).all_accuracies];
            end
        end
        boxplot(cv_data_matrix);
        set(gca, 'XTickLabel', successful_model_names);
        title('交叉验证结果分布');
        ylabel('准确率');
        xtickangle(45);
        
        % 子图2: 交叉验证稳定性
        subplot(2, 2, 2);
        bar(success_cv_stds, 'FaceColor', [0.8 0.3 0.3]);
        set(gca, 'XTickLabel', successful_model_names);
        title('模型稳定性（标准差）');
        ylabel('标准差');
        xtickangle(45);
        
        % 子图3: 平均性能vs稳定性
        subplot(2, 2, 3);
        scatter(success_cv_stds, success_cv_means, 100, 'filled');
        for i = 1:length(successful_models)
            text(success_cv_stds(i) + 0.001, success_cv_means(i), ...
                 successful_model_names{i}, 'HorizontalAlignment', 'left');
        end
        xlabel('标准差（不稳定性）');
        ylabel('平均准确率');
        title('性能vs稳定性权衡');
        grid on;
        
        % 子图4: 各折次结果对比
        subplot(2, 2, 4);
        plot(cv_data_matrix, 'o-', 'LineWidth', 2);
        xlabel('交叉验证折次');
        ylabel('准确率');
        title('各折次结果对比');
        legend(successful_model_names, 'Location', 'best');
        grid on;
        
        sgtitle('图8: 交叉验证详细分析', 'FontSize', 16, 'FontWeight', 'bold');
        saveas(gcf, fullfile(result_dir, '图8_交叉验证详细分析.png'));
    end
    
    fprintf('已生成8张详细的可视化图表，保存到 %s 文件夹\n', result_dir);
else
    fprintf('没有成功训练的模型，跳过可视化\n');
end

%% 7. 生成分析报告
fprintf('\n=== 生成分析报告 ===\n');

report_file = fullfile(result_dir, '源域故障诊断分析报告.txt');
fid = fopen(report_file, 'w', 'n', 'UTF-8');

fprintf(fid, '高速列车轴承智能故障诊断 - 源域故障诊断分析报告\n');
fprintf(fid, '生成时间: %s\n\n', datestr(now));

fprintf(fid, '=== 数据概况 ===\n');
fprintf(fid, '总样本数: %d\n', length(labels));
fprintf(fid, '最终特征维度: %d\n', size(feature_matrix_normalized, 2));
fprintf(fid, '训练集样本数: %d\n', size(X_train, 1));
fprintf(fid, '测试集样本数: %d\n', size(X_test, 1));

fprintf(fid, '\n=== 模型训练结果 ===\n');
fprintf(fid, '成功训练的模型数量: %d/%d\n', length(successful_models), length(models));

for i = 1:length(successful_models)
    model_name = successful_models{i};
    display_name = successful_model_names{i};
    fprintf(fid, '\n%s:\n', display_name);
    fprintf(fid, '  测试准确率: %.4f\n', model_results.(model_name).accuracy);
    fprintf(fid, '  训练时间: %.2f秒\n', model_results.(model_name).training_time);
    if isfield(cv_results, model_name)
        fprintf(fid, '  交叉验证准确率: %.4f ± %.4f\n', ...
            cv_results.(model_name).mean_accuracy, cv_results.(model_name).std_accuracy);
    end
end

fprintf(fid, '\n=== 分析结论 ===\n');
if ~isempty(successful_models)
    [~, best_idx] = max(success_accuracies);
    fprintf(fid, '最佳模型: %s (准确率: %.4f)\n', successful_model_names{best_idx}, success_accuracies(best_idx));
    
    fprintf(fid, '\n建议:\n');
    fprintf(fid, '1. 数据预处理对朴素贝叶斯模型特别重要，建议使用kernel分布\n');
    fprintf(fid, '2. 特征工程可以进一步优化模型性能\n');
    fprintf(fid, '3. 可以考虑集成学习方法提高整体性能\n');
else
    fprintf(fid, '所有模型训练失败，建议检查数据质量和特征工程\n');
end

fclose(fid);

fprintf('分析报告已保存到: %s\n', report_file);
fprintf('\n=== 源域故障诊断任务完成 ===\n');