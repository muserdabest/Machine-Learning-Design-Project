
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
from pathlib import Path 
import scipy.io as sio
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd



def save (errors) -> None:
   import numpy as np
   from pathlib import Path

   arr = np.array(errors)
   if len( arr.shape) > 2 or ( len(arr.shape) == 2 and 1 not in arr.shape):
     raise ValueError(
       "Invalid output shape. Output should be an array "
       "that can be unambiguously raveled/squeezed."
     )
   if arr. dtype not in [np. float64, np.float32, np.float16]:
     raise ValueError( "Your error rates must be stored as float values.")
   arr = arr.ravel()
   if len( arr) != 20 or (arr[ 0] >= arr[ -1]):
     raise ValueError(
       "There should be 20 error values, with the first value "
       "corresponding to k=1, and the last to k=20."
     )
   if arr[ -1] >= 2.0:
     raise ValueError(
       "Final array value too large. You have done something "
       "very wrong (probably relating to standardizing)."
     )
   if arr[ -1] < 0.8:
     raise ValueError(
       "You probably have not converted your error rates to percent values."
     )
   outfile = Path(__file__). resolve().parent / "errors.npy"
   np.save( outfile, arr, allow_pickle=False)
   print(f"Error rates succesfully saved to {outfile }")
def question1():

    mat_data = sio.loadmat('NumberRecognitionAssignment1.mat')
    X_train_8_9 = np.concatenate((mat_data['imageArrayTraining8'],mat_data['imageArrayTraining9']), axis=2)
    y_train_8_9 = np.concatenate((np.ones(np.shape(mat_data['imageArrayTraining8'])[-1]), np.zeros(np.shape(mat_data['imageArrayTraining9'])[-1])))

    X_test_8_9 = np.concatenate((mat_data['imageArrayTesting8'],mat_data['imageArrayTesting9']), axis=2)
    y_test_8_9 = np.concatenate((np.ones(np.shape(mat_data['imageArrayTesting8'])[-1]), np.zeros(np.shape(mat_data['imageArrayTesting9'])[-1])))



    X_train_8_9 = np.reshape(X_train_8_9, (X_train_8_9.shape[-1], X_train_8_9.shape[0]*X_train_8_9.shape[1]))
    X_test_8_9 = np.reshape(X_test_8_9, (X_test_8_9.shape[-1], X_test_8_9.shape[0]*X_test_8_9.shape[1]))


    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_8_9 = scaler.fit_transform(X_train_8_9)
    X_test_8_9 = scaler.transform(X_test_8_9)


    k_range = range(1,21)
    test_error_rate = []
    test_error_rate2 = []
    auc_list = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_8_9, y_train_8_9)
        train_score = knn.score(X_train_8_9, y_train_8_9)
        y_pred = knn.predict(X_test_8_9)
        print(100-train_score**(1/32)*100)
        test_error_rate.append(100 - train_score**(1/32)*100)
        test_error_rate2.append(100-knn.score(X_test_8_9, y_test_8_9)**(1/32)*100)
        y_pred_prob = knn.predict_proba(X_test_8_9)[:,1]
        auc = roc_auc_score(y_test_8_9, y_pred_prob)
        auc_list.append(auc)
    save(test_error_rate)



    #visualization

    plt.plot(k_range, test_error_rate)
    plt.suptitle("in the train set")
    plt.xlabel('K')
    plt.ylabel('test error rate')
    plt.show()

    plt.plot(k_range, test_error_rate2)
    plt.suptitle("in the test set")
    plt.xlabel('K')
    plt.ylabel('test error rate')
    plt.show()

    plt.plot(k_range, auc_list)
    plt.suptitle("The change of AUC value")
    plt.xlabel('K')
    plt.ylabel('AUC')
    plt.show()


 





# ### question 2

def question2():



    adata=pd.read_csv('adult_data.csv', header=None)




    adata




    adata[14].value_counts()




    adata.isnull().any() #No missing value




    adata.info() #Check which variables have 'object' types




    #View non-numeric value information
    non_numerical_columns = [1,3,5,6,7,8,9,13,14]
    for col in non_numerical_columns:
        print(col, adata[col].unique())




    #label coding
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    for col in non_numerical_columns:
    #     print(adata[col])
        encoder.fit(adata[col])
        print(encoder.classes_)
        adata[col] = encoder.transform(adata[col])




    adata




    adata.columns=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain',
                   'capital-loss','hours-per-week','native-country','income']




    #Split data set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(adata.drop('income', axis=1), adata['income'], test_size=0.2, random_state=42)




    #the auc value of each variable by logistic regression
    from sklearn.linear_model import LogisticRegression

    auc_values = {}

    # The auc value of each variable was calculated by logistic regression model
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
    for feature, auc in top_10.items():
        print(feature, ":", auc)




    #Visualize with a bar chart
    plt.barh(list(top_10.keys()), top_10.values())
    plt.suptitle("Top 10 features with highest AUC")
    plt.ylabel("Feature")
    plt.xlabel("AUC")

    plt.show()


# ### question 3

def question3():
    adata = pd.read_csv('adult_data.csv', header=None)

    adata

    adata[14].value_counts()

    adata.isnull().any()  # No missing value

    adata.info()  # Check which variables have 'object' types

    # View non-numeric value information
    non_numerical_columns = [1, 3, 5, 6, 7, 8, 9, 13, 14]
    for col in non_numerical_columns:
        print(col, adata[col].unique())

    # label coding
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    for col in non_numerical_columns:
        #     print(adata[col])
        encoder.fit(adata[col])
        print(encoder.classes_)
        adata[col] = encoder.transform(adata[col])

    adata

    adata.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                     'relationship', 'race', 'sex', 'capital-gain',
                     'capital-loss', 'hours-per-week', 'native-country', 'income']

    # Split data set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(adata.drop('income', axis=1), adata['income'], test_size=0.2,
                                                        random_state=42)


    k_range = range(1,21)
    test_error_rate = []
    test_error_rate2 = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        train_score = knn.score(X_train, y_train)
        y_pred = knn.predict(X_test)
        print(1 - train_score)
        test_error_rate.append(1 - train_score)
        test_error_rate2.append(1-accuracy_score(y_pred,y_test))

    #visualization
    plt.plot(k_range, test_error_rate)
    plt.suptitle("in the train set")
    plt.xlabel('K')
    plt.ylabel('test error rate')
    plt.show()

    plt.plot(k_range, test_error_rate2)
    plt.suptitle("in the test set")
    plt.xlabel('K')
    plt.ylabel('test error rate')
    plt.show()


if __name__ == "__main__":
   # setup / helper function calls here, if using
   question1()
   question2() # these functions can optionally take arguments (e.g. `Path`s to your data)
   question3()




