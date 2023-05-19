import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def give_feature_importances(model, columns_name,  rotation_angel=45):
    feature_importances = pd.DataFrame(
        {
            'feature_name' : columns_name,
            'importance': model.feature_importances_
        }
    )
    sns.barplot(data=feature_importances,x='feature_name',y='importance')
    plt.xticks(rotation=rotation_angel)
    plt.show()

from sklearn.metrics import accuracy_score, classification_report
def give_acc_report(y_test, y_predict, target_names=None):
    print('accuracy ', accuracy_score(y_true=y_test, y_pred=y_predict))
    if target_names:
        print(classification_report(
            y_true=y_test,
            y_pred=y_predict,
            target_names=target_names
        ))
    else:
        print(classification_report(
            y_true=y_test,
            y_pred=y_predict,
        ))

from sklearn.model_selection import learning_curve, LearningCurveDisplay

def give_learning_curve(model, X_train, y_train,  train_sizes= np.linspace(0.1,1,10),cv=5):

    params = {
        'estimator': model,
        'X': X_train,
        'y': y_train,
        'train_sizes': train_sizes,
        'cv': cv,
        'score_type':'both',
        "score_name": "Accuracy",
        "line_kw": {"marker": "o"},
        }
    LearningCurveDisplay.from_estimator(**params)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
def get_confusion_matrix(y_test, y_predict, labels=None):
    sns.set_style("white")
    conf_mat = confusion_matrix(y_true=y_test, 
                                y_pred=y_predict)
    if labels:
        ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=labels).plot()
    else:
        ConfusionMatrixDisplay(confusion_matrix=conf_mat).plot()

def show_all(model, X_train, y_train, y_test, y_pred, columns_name):
    give_feature_importances(model, columns_name)
    give_acc_report(y_test, y_pred)
    give_learning_curve(model, X_train, y_train)
    get_confusion_matrix(y_test, y_pred)