import pandas as pd
import numpy as np

#missing value
def check_and_remove_missing_values(df, dataset_name):
    # 每個欄位缺失值數量
    missing_per_column = df.isnull().sum()

    # 總缺失值數量
    total_missing = missing_per_column.sum()

    # 含有缺失值的資料筆數
    missing_rows = df.isnull().any(axis=1).sum()


    if missing_rows > 0:
        original_rows = len(df) #原本資料筆數
        df = df.dropna() #刪除有缺失值之資料
        removed_rows = original_rows - len(df) #刪除資料筆數

        #print(f"\nMissing values found.")
        #print(f"Removed {removed_rows} rows.")
        #print(f"Remaining rows: {len(df)}")
    #else:
        #print("No missing values found.")
        #print("No rows removed.")

    return df


# 1. Load datasets
banknote = pd.read_csv(
    "banknote+authentication/data_banknote_authentication.txt",
    header=None
)

glass = pd.read_csv(
    "glass+identification/glass.data",
    header=None
)

segmentation_train = pd.read_csv(
    "image+segmentation/segmentation.data",
    skiprows=4,
    header=None
)

segmentation_test = pd.read_csv(
    "image+segmentation/segmentation.test",
    skiprows=4,
    header=None
)

segmentation = pd.concat(
    [segmentation_train, segmentation_test],
    ignore_index=True
)

# 2. Set column names
banknote.columns = ["variance", "skewness", "curtosis", "entropy", "class"]

glass.columns = ["id", "RI", "Na", "Mg", "Al", "Si","K", "Ca", "Ba", "Fe", "class"]

segmentation.columns = ["class","REGION-CENTROID-COL","REGION-CENTROID-ROW","REGION-PIXEL-COUNT","SHORT-LINE-DENSITY-5",
    "SHORT-LINE-DENSITY-2","VEDGE-MEAN","VEDGE-SD","HEDGE-MEAN","HEDGE-SD","INTENSITY-MEAN","RAWRED-MEAN","RAWBLUE-MEAN","RAWGREEN-MEAN",
    "EXRED-MEAN","EXBLUE-MEAN","EXGREEN-MEAN","VALUE-MEAN","SATURATION-MEAN","HUE-MEAN"]

# 3. Check missing values
banknote = check_and_remove_missing_values(
    banknote,
    "Banknote Authentication"
)

glass = check_and_remove_missing_values(
    glass,
    "Glass Identification"
)

segmentation = check_and_remove_missing_values(
    segmentation,
    "Image Segmentation"
)

#X為屬性欄位 Y為類別欄位
X_banknote = banknote.iloc[:, :-1]
y_banknote = banknote.iloc[:, -1]

X_glass = glass.iloc[:, 1:-1]
y_glass = glass.iloc[:, -1]

X_seg = segmentation.iloc[:, 1:]
y_seg = segmentation.iloc[:, 0]


#資料離散化
def discretize_equal_width(X, bins=10):
    X_discrete = X.copy()
    for col in X.columns:
        min_value = X[col].min()
        max_value = X[col].max()
        width = (max_value - min_value) / bins
        bin_edges = []
        for i in range(bins + 1):
            bin_edges.append(min_value + i * width) #桶子邊界
        new_values = []
        for value in X[col]:
            bucket = bins - 1
            for j in range(len(bin_edges)-1):  #分配到桶子
                if bin_edges[j] <= value < bin_edges[j+1]:
                    bucket = j
                    break
            new_values.append(bucket) #存放桶子
        X_discrete[col] = new_values #離散化後的
    return X_discrete

X_banknote_discrete = discretize_equal_width(X_banknote)
X_glass_discrete = discretize_equal_width(X_glass)
X_seg_discrete = discretize_equal_width(X_seg)
#print(X_glass_discrete) #測試離散化後的資料


#Naive Bayesian Classifier
#step 1 P(Cj)
def calculate_prior(y_train):
    priors = {}
    total_count = len(y_train)
    classes = y_train.unique()
    for c in classes:
        class_count = len(y_train[y_train == c])
        priors[c] = class_count / total_count
    return priors

#step 2 P(Xi|Cj) 
#def calculate_conditional_probability(X_train,y_train,column,value,target_class, bins=10):
#    class_rows = y_train[y_train == target_class].index  #先找哪幾筆為target_class
#    class_count = len(class_rows) #計算target_class筆數
#    count = 0
#    for idx in class_rows:
#        if X_train.loc[idx, column] == value:  #分子 這邊的value是桶子(離散化後的)
#            count += 1
#    return (count+1) / (class_count + bins) #Laplace

#step 3 NBC
#def predict_one(X_row,X_train,y_train,priors):
#    best_class = None #用來記錄分數最高之類別
#    best_score = -1 #用來計算目前最高分數
#    classes = y_train.unique() #取得所有類別
#    for c in classes:
#        score = priors[c]
#       for col in X_train.columns:
#            value = X_row[col]
#            conditional_prob = calculate_conditional_probability(X_train,y_train,col,value,c)
#            score *= conditional_prob
#        if score > best_score: #選擇較大機率的 紀錄他的機率值跟類別
#            best_score = score
#            best_class = c
#    return best_class #回傳預測類別


#predict
#def predict(X_test, X_train, y_train, priors):
#    predictions = []
#    for i in range(len(X_test)):
#        X_row = X_test.iloc[i]
#        predicted_class = predict_one(X_row,X_train,y_train,priors)
#        predictions.append(predicted_class)
#    return predictions

#上述部份執行太慢，改寫成下面：
# train NBC：先建立機率表
def train_nbc(X_train, y_train, bins=10):
    priors = calculate_prior(y_train)
    classes = y_train.unique()
    conditional_probs = {}

    for c in classes:
        conditional_probs[c] = {}
        class_rows = y_train[y_train == c].index
        class_count = len(class_rows)

        for col in X_train.columns:
            conditional_probs[c][col] = {} 

            for value in range(bins):
                count = 0

                for idx in class_rows:
                    if X_train.loc[idx, col] == value:
                        count += 1
                conditional_probs[c][col][value] = (count + 1) / (class_count + bins)  #laplace
    return priors, conditional_probs

# predict one row：直接查機率表
def predict_one_fast(X_row, priors, conditional_probs):
    best_class = None
    best_score = -1

    for c in priors:
        score = priors[c]

        for col in X_row.index:
            value = X_row[col]
            score *= conditional_probs[c][col][value]

        if score > best_score:
            best_score = score
            best_class = c
    return best_class

# predict whole test set
def predict_fast(X_test, priors, conditional_probs):
    predictions = []
    for i in range(len(X_test)):
        X_row = X_test.iloc[i]
        predicted_class = predict_one_fast(X_row, priors, conditional_probs)
        predictions.append(predicted_class)
    return predictions

#accuracy
def calculate_accuracy_info(y_true, y_pred):
    correct = 0

    for i in range(len(y_true)):
        if y_true.iloc[i] == y_pred[i]:
            correct += 1
    total = len(y_true)
    accuracy = correct / total

    return accuracy, correct, total

#k-fold
def create_folds(X, k=5):
    indices = list(range(len(X)))
    np.random.shuffle(indices)
    base_size = len(X) // k
    remainder = len(X) % k
    folds = []
    start = 0
    for i in range(k):
        #前remainder個fold多分一筆
        if i < remainder:
            current_size = base_size + 1
        else:
            current_size = base_size
        end = start + current_size
        folds.append(indices[start:end])
        start = end
    return folds

#NBC 5-fold 
def evaluate_nbc_5fold(X, y, folds):
    fold_accuracies = []
    fold_correct = []
    fold_sizes = []

    for i in range(len(folds)):
        print("Running NBC Fold", i+1)
        test_index = folds[i]
        train_index = []
        for j in range(len(folds)):
            if j != i:
                train_index.extend(folds[j])

        X_train = X.iloc[train_index].reset_index(drop=True)
        y_train = y.iloc[train_index].reset_index(drop=True)
        X_test = X.iloc[test_index].reset_index(drop=True)
        y_test = y.iloc[test_index].reset_index(drop=True)

        priors, conditional_probs = train_nbc(X_train, y_train)
        predictions = predict_fast(X_test, priors, conditional_probs)
        acc, correct, total = calculate_accuracy_info(y_test, predictions)

        fold_accuracies.append(acc)
        fold_correct.append(correct)
        fold_sizes.append(total)
    return {
        "fold_accuracies": fold_accuracies,
        "fold_correct": fold_correct,
        "fold_sizes": fold_sizes,
        "fold_average_accuracy": sum(fold_accuracies) / len(fold_accuracies),
        "total_correct": sum(fold_correct),
        "total_size": sum(fold_sizes),
        "dataset_accuracy": sum(fold_correct) / sum(fold_sizes)
    }




#bagging
#bootstrap sampling
def bootstrap_sample(X_train, y_train):
    sample_indices = []
    for i in range(len(X_train)): #放回抽
        random_index = np.random.randint(0, len(X_train))
        sample_indices.append(random_index)
    X_sample = X_train.iloc[sample_indices].reset_index(drop=True)
    y_sample = y_train.iloc[sample_indices].reset_index(drop=True)
    return X_sample, y_sample

#bagging predict
def bagging_predict(X_test, X_train, y_train, n_models):
    all_model_predictions = []
    for m in range(n_models): #n個模型
        X_sample, y_sample = bootstrap_sample(X_train, y_train)
        priors, conditional_probs = train_nbc(X_sample, y_sample)
        predictions = predict_fast(X_test, priors, conditional_probs)
        all_model_predictions.append(predictions)
    final_predictions = []

    for i in range(len(X_test)): #投票
        votes = {}
        for m in range(n_models): 
            predicted_class = all_model_predictions[m][i]
            if predicted_class not in votes:
                votes[predicted_class] = 1
            else:
                votes[predicted_class] += 1
        best_class = max(votes, key=votes.get)
        final_predictions.append(best_class)
    return final_predictions

#Bagging 5-fold evaluation
def evaluate_bagging_5fold(X, y, folds, n_models):
    fold_accuracies = []
    fold_correct = []
    fold_sizes = []

    for i in range(len(folds)):
        print(f"Running Bagging({n_models}) Fold", i+1)
        test_index = folds[i]
        train_index = []
        for j in range(len(folds)):
            if j != i:
                train_index.extend(folds[j])

        X_train = X.iloc[train_index].reset_index(drop=True)
        y_train = y.iloc[train_index].reset_index(drop=True)
        X_test = X.iloc[test_index].reset_index(drop=True)
        y_test = y.iloc[test_index].reset_index(drop=True)

        predictions = bagging_predict(X_test, X_train, y_train, n_models)
        acc, correct, total = calculate_accuracy_info(y_test, predictions)

        fold_accuracies.append(acc)
        fold_correct.append(correct)
        fold_sizes.append(total)
    return {
        "fold_accuracies": fold_accuracies,
        "fold_correct": fold_correct,
        "fold_sizes": fold_sizes,
        "fold_average_accuracy": sum(fold_accuracies) / len(fold_accuracies),
        "total_correct": sum(fold_correct),
        "total_size": sum(fold_sizes),
        "dataset_accuracy": sum(fold_correct) / sum(fold_sizes)
    }


#執行部分
datasets = [("Banknote", X_banknote_discrete, y_banknote),("Glass", X_glass_discrete, y_glass),("Segmentation", X_seg_discrete, y_seg)]
results = {}
for dataset_name, X_data, y_data in datasets:
    print("\n==============================")
    print(dataset_name)
    print("==============================")

    folds = create_folds(X_data, k=5)

    results[dataset_name] = {}

    results[dataset_name]["NBC"] = evaluate_nbc_5fold(X_data,y_data,folds)

    for n_models in [10, 20, 30]:
        method_name = f"Bagging({n_models})"

        results[dataset_name][method_name] = evaluate_bagging_5fold(X_data,y_data,folds,n_models)

    for method_name, result in results[dataset_name].items():
        print("\nMethod:", method_name)
        print("Fold Accuracies:", result["fold_accuracies"])
        print("Fold Correct:", result["fold_correct"])
        print("Fold Sizes:", result["fold_sizes"])
        print("Fold Average Accuracy:", result["fold_average_accuracy"])
        print("Dataset Accuracy:", result["dataset_accuracy"])
        print("Total Correct:", result["total_correct"])
        print("Total Size:", result["total_size"])