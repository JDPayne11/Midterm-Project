import numpy as np
from sklearn import metrics
# from sklearn.preprocessing import StandardScaler

def scale_features(X_train, X_test, numeric_cols, cat_cols, scaler):
    '''returns X_train and X_test array where the numerical columns are scaled,
    then combines it with the categorical array'''
    scaler = scaler.fit(X_train[numeric_cols])

    X_train_scaled = scaler.transform(X_train[numeric_cols])
    X_test_scaled = scaler.transform(X_test[numeric_cols])

    # combine with categorical columns
    X_train_scaled_comb = np.hstack((X_train_scaled, X_train[cat_cols]))
    X_test_scaled_comb = np.hstack((X_test_scaled, X_test[cat_cols]))

    return X_train_scaled_comb, X_test_scaled_comb


def regression_metrics(y_true, yhat):
    '''return R^2, MSE, MAE, and RMSE metrics'''

    print(f'R^2: {metrics.r2_score(y_true, yhat)}')
    print(f'Mean Squared Error: {metrics.mean_squared_error(y_true, yhat)}')
    print(f'Mean Absolute Error: {metrics.mean_absolute_error(y_true, yhat)}')
    print(f'Root MSE: {metrics.mean_squared_error(y_true, yhat, squared=False)}')