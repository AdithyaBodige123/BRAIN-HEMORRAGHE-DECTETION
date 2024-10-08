from sklearn.tree import DecisionTreeClassifier
import Code1.Draw


def decisionTreeClassifier(X_train, y_train, X_test, y_test, images, index):
    DtreeClf = DecisionTreeClassifier()
    DtreeClf.fit(X_train, y_train)
    acc = DtreeClf.score(X_test, y_test) * 100
    Code1.Draw.drawPredict(DtreeClf, X_test, y_test, images, index)
    return acc
