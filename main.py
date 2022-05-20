from util import *


def model_creation():
    """
    This function helps in creating ml model
    :return: model
    """

    # creating object
    data = ConstantValue()
    final_data = data.data_preparation(data.file_path)
    cv = CountVectorizer(max_features=data.max_features_value, encoding="utf-8",
                         ngram_range=(data.one_value, data.three_value),
                         token_pattern="[A-Za-z_][A-Za-z\d_]*")
    X = cv.fit_transform(final_data.all_features).toarray()
    y = final_data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=data.test_size_value,
                                                        random_state=data.random_state_value, stratify=y)
    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    matrix = classification_report(y_test, y_pred)
    print('Classification report : \n', matrix)
    return matrix


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model_evaluation = model_creation()
    print("<<<<<<<The model is Evaluated>>>>>>>>>>>>>")
