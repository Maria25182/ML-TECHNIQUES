import pandas as pd

from sklearn.cluster import MiniBatchKMeans

if __name__ == "__main__":
    
    
    dataset = pd.read_csv("../data/candy.csv")
    print (dataset.head(10))

''' x_train, x_test, y_train, y_test'''

##NO SUPERVISADO

x= dataset.drop('competitorname',axis=1)


kmeans = MiniBatchKMeans(n_clusters=4,batch_size=8).fit(x)
print("total de centros : ",len(kmeans.cluster_centers_))
print("="*64)
print(kmeans.predict(x))

dataset['group'] = kmeans.predict(x)

print(dataset)