import pandas as pd 
import sklearn
import matplotlib.pyplot as plt



from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression  ##Principal function is classification
from sklearn.preprocessing import StandardScaler ## all data scale out 0 and 1
from sklearn.model_selection import train_test_split ##



if __name__ == "__main__":
    df_heart = pd.read_csv("../data/heart.csv")
    ##print(df_heart)
    
    df_features = df_heart.drop(['target'],axis=1)
    df_target = df_heart['target']
    
    df_features = StandardScaler().fit_transform(df_features)
    
    x_train ,x_test, y_train,y_test = train_test_split(df_features,df_target,test_size=0.3,random_state=42)
    
    print(x_train.shape)

    print(y_train.shape)
    
    '''
    ipca es un objeto de IncrementalPCA, que es un método de reducción de dimensionalidad. Cuando usas fit en IncrementalPCA,
    estás calculando los componentes principales del conjunto de datos x_train para reducir su dimensionalidad.
    Aquí, solo necesitas el conjunto de características x_train para calcular los componentes principales que describen la variabilidad de los datos.
    '''
    pca = PCA(n_components=3)
    pca.fit(x_train)
    
    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(x_train)
    

    plt.plot(range(len(pca.explained_variance_)),pca.explained_variance_ratio_)
    ##plt.show()
    
    
    '''Logistic.fit(dt_train, y_train):
    logistic es un modelo de regresión logística, que es un modelo de aprendizaje supervisado.
    El método fit de LogisticRegression ajusta el modelo a los datos proporcionados para aprender a hacer predicciones.
    Necesitas dos conjuntos de datos:
    dt_train: Las características transformadas por PCA que se utilizarán para entrenar el modelo.
    y_train: Las etiquetas correspondientes a dt_train que el modelo intentará predecir.'''
    logistic = LogisticRegression (solver='lbfgs')
    
    
   # Configuramos los datos de entrenamiento
    dt_train = pca.transform(x_train)
    dt_test = pca.transform(x_test)
    
    
  # Mandamos los data frames la la regresión logística
    logistic.fit(dt_train, y_train)
  #Calculamos nuestra exactitud de nuestra predicción
    print("SCORE PCA: ", logistic.score(dt_test, y_test))
    
    
  #Configuramos los datos de entrenamiento
    dt_train = ipca.transform(x_train)
    dt_test = ipca.transform(x_test)
    
    
  # Mandamos los data frames la la regresión logística
    logistic.fit(dt_train, y_train)
  #Calculamos nuestra exactitud de nuestra predicción
    print("SCORE IPCA: ", logistic.score(dt_test, y_test))