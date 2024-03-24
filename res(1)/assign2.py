import scipy.io as sio
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

def load_data(path):
    mat_data = sio.loadmat(path)
    print(mat_data["__header__"])
    print(mat_data["__version__"])
    print("images:", mat_data["X"].shape)
    print("labels:", mat_data["y"].shape, "\n")

    print("get 8 and 9 from the original dataset, keep the original order")
    X89 = []
    y89 = []
    X = np.transpose(mat_data["X"],(2,0,1))
    y = mat_data["y"][0]
    for i, item in enumerate(y):
        if item == 8 or item == 9:
            X89.append(X[i])
            y89.append(item)
    print("the nums of 8 and 9 is:", len(y89), "\n")

    print("choose a image randomly to show")
    order = random.randint(0,len(y89)-1)
    print("order", order)
    plt.imshow(X89[order])
    plt.title("the order "+str(order)+" image should be:"+str(y89[order])+". right?")
    plt.draw()
    plt.pause(3)
    plt.close()

    return X89, y89


def Question1():
    print("load data...")
    X, y = load_data(r'NumberRecognitionBigger.mat')
    X = np.array(X)
    X = X.reshape((5870,784))
    y = np.array(y)
    print(X.shape)
    print(y.shape)

    print("create the classifier")
    svm_linear = SVC(kernel="linear")
    svm_RBF = SVC(kernel="rbf")
    rf_100 = RandomForestClassifier(n_estimators=100)
    knn_1 = KNeighborsClassifier(n_neighbors=1)
    knn_5 = KNeighborsClassifier(n_neighbors=5)
    knn_10 = KNeighborsClassifier(n_neighbors=10)

    print("running...")
    #the k-fold is fixed so the result can be repeated
    names = ["svm_linear", "svm_RBF", "rf_100", "knn_1", "knn_5", "knn_10"]
    best = 0
    best_classifier = ""
    s = []
    for i,item in enumerate([svm_linear, svm_RBF, rf_100, knn_1, knn_5, knn_10]):
        scores = cross_val_score(item, X, y, cv=5)
        s.append(scores)
        score = max(scores)
        print("The scores of "+names[i]+" about 5-fold cross validation:")
        print(scores)
        print("The best score of "+names[i]+"about 5-fold cross validation:")
        print(score, "\n")
        if score>best:
            best = score
            best_classifier = names[i]
    print("The best score of all classifier about 5-fold cross validation:")
    print(best_classifier, best)
    print(s)
    #write the json
    m = []
    for item in s:
        m.append(sum(item)/5)
    res = []
    for i in range(6):
        cur = np.append(s[i],m[i])
        res.append(cur)
    res = np.array(res)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i][j]=1-res[i][j]
    cols = [*[f"fold{i}" for i in range(1, 6)], "mean"]
    index = ["svm_linear", "svm_rbf", "rf", "knn1", "knn5", "knn10"]
    res_data = pd.DataFrame(res, columns=cols, index=index, dtype=np.float64)
    res_data.to_json("kfold_mnist.json")
    print(res_data.round(3).to_markdown())


def Question2():
    adata=pd.read_csv(r"adult_data.csv", header=None)
    header = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
             'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
             'hours-per-week', 'native-country', 'income']
    print(adata.info(), "\n")
    print("choose a row of data randomly to show")
    order = random.randint(0,adata.shape[0])
    print("order", order)
    print(adata.loc[0, :], "\n")

    print("total number of samples:")
    print(adata.shape[0])
    print("total number of measurements:")
    print(adata.shape[1])
    print("brief description of measurements:")
    print(header)
    print("nature of the group of interest:")
    print(header[-1])
    print("we want to predict income with some measurements like age, education, sex etc")

    print("check if there are some invalid values")
    print(adata.isna().any())
    encoder = LabelEncoder()
    for i in range(adata.shape[1]):
        encoder.fit(adata[i])
        print(i)
        print(encoder.classes_)

    lines_miss_value = set()
    print("we find there are some ？ with the values")
    #print(adata[13][32525])
    for i in ([1, 6, 13]):
        for j, item in enumerate(adata[i]):
            if item == " ?":
                lines_miss_value.add(j)
                #print(j, item)
    print("we count the line which contain miss value:")
    print(len(lines_miss_value))
    print("2399 is quiet smaller than 32561 we decide drop it")
    print(adata.shape[0])
    for i in lines_miss_value:
        adata_1 = adata.drop(labels=lines_miss_value,axis=0)
    print(adata_1.shape[0])

    #quantification
    encoder = LabelEncoder()
    for i in range(adata_1.shape[1]):
        adata_1[i] = encoder.fit_transform(adata_1[i])

    #Split data set
    X_train, X_test, y_train, y_test = train_test_split(adata_1.drop(14, axis=1), adata_1[14], test_size=0.2, random_state=42)

    #the auc value of each variable by logistic regression
    auc_values = {}
    for feature in X_train.columns:
        lr = LogisticRegression()
        lr.fit(X_train[[feature]], y_train)
        y_pred = lr.predict_proba(X_test[[feature]])[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        auc_values[feature] = auc

    #Sort the auc values
    sorted_auc = dict(sorted(auc_values.items(), key=lambda item: item[1], reverse=True))

    #Print the first 10 variables
    top_10 = {k: sorted_auc[k] for k in list(sorted_auc)[:10]}
    print("Top 10 features with highest AUC:")
    key = []
    for feature, auc in top_10.items():
        key.append(feature)
        print(header[feature], ":", auc)
    print(key)
    '''
    plt.rcParams["figure.figsize"] = (8,6)
    plt.bar(list(header[feature] for feature in top_10.keys()), top_10.values(), width=0.5)
    plt.suptitle("Top 10 features with highest AUC")
    plt.ylabel("Feature")
    plt.xlabel("AUC")
    plt.xticks(rotation=270)
    plt.xticks(fontsize=6)
    plt.show()
    '''

    #write the json
    res = []
    for item in sorted_auc:
        res.append([header[item],sorted_auc[item]])
    cols = ["feature", "auc"]
    print(res)
    res_data = pd.DataFrame(np.array(res), columns=cols)
    res_data.to_json("aucs.json")
    print(res_data.round(3).to_markdown(index=False))


def Question3():
    print("load data...")
    adata = pd.read_csv(r"adult_data.csv", header=None)
    lines_miss_value = set()
    for i in ([1, 6, 13]):
        for j, item in enumerate(adata[i]):
            if item == " ?":
                lines_miss_value.add(j)
    for i in lines_miss_value:
        adata_1 = adata.drop(labels=lines_miss_value, axis=0)
    print(adata_1.shape[0])
    encoder = LabelEncoder()
    for i in range(adata_1.shape[1]):
        adata_1[i] = encoder.fit_transform(adata_1[i])

    # Split data set
    X = adata_1.drop(labels=[1, 2, 3, 13, 14], axis=1)#4, 7, 0, 12, 5, 9, 10, 11, 6, 8
    y = adata_1[14]
    X = X.to_numpy()
    y = y.to_numpy()
    count = 0
    for item in y:
        if item==1:
            count+=1
    print(count)
    print(X.shape)
    print(y.shape)
    #for i in range(10):
    #    X[:][i] = X[:][i]/max(X[:][i])
    #print(X[0:5])
    #print(y[0:5])
    X = X[0:10000]
    y = y[0:10000]
    #svm_linear = SVC(kernel="linear", max_iter=1000000)
    #svm_RBF = SVC(kernel="rbf", max_iter=1000000)
    print("create the classifier")
    svm_linear = SVC(kernel="linear")
    svm_RBF = SVC(kernel="rbf")
    rf_100 = RandomForestClassifier(n_estimators=100)
    knn_1 = KNeighborsClassifier(n_neighbors=1)
    knn_5 = KNeighborsClassifier(n_neighbors=5)
    knn_10 = KNeighborsClassifier(n_neighbors=10)

    print("running...")
    # the k-fold is fixed so the result can be repeated
    names = ["svm_linear", "svm_RBF", "rf_100", "knn_1", "knn_5", "knn_10"]
    best = 0
    best_classifier = ""
    s = []
    for i, item in enumerate([svm_linear, svm_RBF, rf_100, knn_1, knn_5, knn_10]):
    #for i, item in enumerate([rf_100, knn_1, knn_5, knn_10]):
        scores = cross_val_score(item, X, y, cv=5)
        s.append(scores)
        score = max(scores)
        print("The scores of " + names[i] + " about 5-fold cross validation:")
        print(scores)
        print("The best score of " + names[i] + "about 5-fold cross validation:")
        print(score, "\n")
        if score > best:
            best = score
            best_classifier = names[i]
    print("The best score of all classifier about 5-fold cross validation:")
    print(best_classifier, best)

    #write the json
    m = []
    for item in s:
        m.append(sum(item)/5)
    res = []
    for i in range(6):
        cur = np.append(s[i],m[i])
        res.append(cur)
    res = np.array(res)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i][j]=1-res[i][j]
    cols = [*[f"fold{i}" for i in range(1, 6)], "mean"]
    index = ["svm_linear", "svm_rbf", "rf", "knn1", "knn5", "knn10"]
    res_data = pd.DataFrame(res, columns=cols, index=index, dtype=np.float64)
    res_data.to_json("kfold_data.json")
    print(res_data.round(3).to_markdown())
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    Question1()
    Question2()
    Question3()

