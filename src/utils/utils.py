from sklearn.datasets import load_breast_cancer, load_diabetes, fetch_20newsgroups, load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from keras.datasets import fashion_mnist
import numpy  as np
def data1(num):
    if num==0:
        data = load_breast_cancer()
    elif num==1:
        data=load_diabetes()
    elif num==2:
        data=fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes')) 
    elif num==3:
        data=load_digits()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train,X_test,y_train, y_test 
def classifier(num,count):
    X_train,X_test,y_train, y_test =data1(num)
    if count==0:
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}")
    elif count==1:
        ab = AdaBoostClassifier()
        ab.fit(X_train, y_train)
        y_pred_ab = ab.predict(X_test)
        print(f"AdaBoost Accuracy: {accuracy_score(y_test, y_pred_ab)}")
    elif count==2:
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(X_train)
        y_pred_kmeans = kmeans.predict(X_test)
        print(f"K-Means Accuracy: {accuracy_score(y_test, y_pred_kmeans)}")
    elif count==3:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        print(f"k-Nearest Neighbors Accuracy: {accuracy_score(y_test, y_pred_knn)}")
    elif count==4:
        regressor = Sequential()
        regressor.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
        regressor.add(Dense(16, activation='relu'))
        regressor.add(Dense(1))
        regressor.compile(optimizer='adam', loss='mean_squared_error')
        regressor.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
        y_pred_dl = regressor.predict(X_test)
        print(f"Deep Learning Regression MSE: {mean_squared_error(y_test, y_pred_dl)}")
    elif count==5 and num==2:
        data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
        texts = data.data
        labels = data.target

        tokenizer = Tokenizer(num_words=10000)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        word_index = tokenizer.word_index

        data = pad_sequences(sequences, maxlen=1000)
        labels = np.asarray(labels)

        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        model = Sequential()
        model.add(Embedding(10000, 32, input_length=1000))
        model.add(LSTM(32))
        model.add(Dense(20, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=15, batch_size=64, verbose=1)
        loss, acc = model.evaluate(X_test,y_test)
        print(f"LSTM Text Classification Accuracy: {acc}")
    elif count==6:
        data = load_digits()
        X, y = data.images, data.target
        X = np.expand_dims(X, axis=-1)
        y = to_categorical(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        input_shape = X_train.shape[1:]
        input_shape = X_train.shape[1:]
        def UNet():
            model = Sequential()
            model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(UpSampling2D(size=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(UpSampling2D(size=(2, 2)))
            model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
            model.add(Conv2D(10, (3, 3), activation='softmax', padding='same'))
            model.add(Flatten())  # Add a Flatten layer here
            model.add(Dense(10, activation='softmax'))
            return model
        unet = UNet()
        unet.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        unet.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
        loss, acc = unet.evaluate(X_test, y_test, verbose=1)
        print(f"UNet Object Localization Accuracy: {acc}")
def auc():
    data = load_breast_cancer()
    X_train,X_test,y_train, y_test =data1(0)
    auc_scores = []
    for i in range(X_train.shape[1]):
        clf = LogisticRegression(random_state=42)
        clf.fit(X_train[:, i].reshape(-1, 1), y_train)
        y_pred = clf.predict_proba(X_test[:, i].reshape(-1, 1))[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        auc_scores.append((data.feature_names[i], auc))
    top_10_measurements = sorted(auc_scores, key=lambda x: x[1], reverse=True)[:10]
    for i, (measurement, auc) in enumerate(top_10_measurements):
        print(f"{i + 1}. {measurement}: AUC = {auc:.4f}")
def hidden_layers():
    X_train,X_test,y_train, y_test =data1(0)
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    for n_layers in [1, 2, 3]:
        model = Sequential()
        model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
        for _ in range(n_layers - 1):
            model.add(Dense(32, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train_cat, epochs=10, batch_size=32, verbose=0)
        loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
        print(f"Deep Learning model with {n_layers} hidden layers: {acc:.4f}")
def rf_es_max():
    X_train,X_test,y_train, y_test =data1(0) 
    for n_estimators in [10, 50, 100]:
        for max_depth in [None, 5, 10]:
            rf_clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            scores = cross_val_score(rf_clf, X_train, y_train, cv=5, scoring='accuracy')
            print(f"Random Forest with {n_estimators} estimators and max_depth {max_depth}: {np.mean(scores):.4f}")
def ada_es():
    X_train,X_test,y_train, y_test =data1(0) 
    for depth in [1, 2, 3]:
        base_learner = DecisionTreeClassifier(max_depth=depth)
        ada_clf = AdaBoostClassifier(base_estimator=base_learner, n_estimators=100, random_state=42)
        scores = cross_val_score(ada_clf, X_train, y_train, cv=5, scoring='accuracy')
        print(f"AdaBoost with Decision Tree depth {depth}: {np.mean(scores):.4f}")
def compare():
    X_train,X_test,y_train, y_test =data1(0)
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_train)
    y_pred_train = kmeans.predict(X_train)
    y_pred_test = kmeans.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"K-Means Unsupervised Learning - Training Accuracy: {train_acc:.4f}")
    print(f"K-Means Unsupervised Learning - Test Accuracy: {test_acc:.4f}")

    #Compare with a supervised learning model (Random Forest)
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    rf_train_acc = rf_clf.score(X_train, y_train)
    rf_test_acc = rf_clf.score(X_test, y_test)

    print(f"Random Forest Supervised Learning - Training Accuracy: {rf_train_acc:.4f}")
    print(f"Random Forest Supervised Learning - Test Accuracy: {rf_test_acc:.4f}")
def epoch_100():
    X_train, X_test, y_train, y_test=data1(1)
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mean_absolute_error'])
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Deep Learning Regression Model - Mean Absolute Error: {mae:.4f}")
def cnn():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train_cat, epochs=10, batch_size=32, verbose=0)
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"CNN Image Classification Accuracy: {acc:.4f}")

