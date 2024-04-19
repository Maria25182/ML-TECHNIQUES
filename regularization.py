import pandas as pd
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet


if __name__ == "__main__":
    dataset = pd.read_csv("../data/felicidad.csv")
    print(dataset.describe())

    x = dataset[
        ["gdp", "family", "lifexp", "freedom", "corruption", "generosity", "dystopia"]
    ]
    y = dataset[["score"]]

    print(x.shape)
    print(y.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    modeLinear = LinearRegression().fit(x_train, y_train)
    y_predict_linear = modeLinear.predict(x_test)
    modelLaso = Lasso(alpha=0.02).fit(x_train, y_train)
    y_predict_lasso = modelLaso.predict(x_test)

    modelRidge = Ridge(alpha=1).fit(x_train, y_train)
    y_predict_ridge = modelRidge.predict(x_test)

    regr = ElasticNet(random_state=0).fit(x, y)
    y_predict_elastic = regr.predict(x_test)

    linear_loss = mean_squared_error(y_test, y_predict_linear)
    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    elastic_loss = mean_squared_error(y_test, y_predict_elastic)

    print(
        f" LOSS  Linear {linear_loss} lasso {lasso_loss} ridge {ridge_loss} Elastic {elastic_loss}"
    )

    # Imprimimos las coficientes para ver como afecta a cada una de las regresiones
    # La lines "="*32 lo unico que hara es repetirme si simbolo de igual 32 veces
    print("=" * 32)
    print("Coeficientes lasso: ")
    # Esta informacion la podemos encontrar en la variable coef_
    print(modelLaso.coef_)

    # Hacemos lo mismo con ridge
    print("=" * 32)
    print("Coeficientes ridge:")
    print(modelRidge.coef_)

    ##
