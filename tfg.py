import csv
import numpy as np
import keras
import tensorflow as tf
import tensorflow_addons as tfa
import sklearn.metrics
from keras import Model
from keras import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from imblearn.ensemble import BalancedRandomForestClassifier
from concrete_autoencoder import ConcreteAutoencoderFeatureSelector

###############################################################################

def open_csv(name : str):
	file = open(name)
	csvreader = csv.reader(file)

	header = next(csvreader)
	rows = []
	for row in csvreader:
		rows.append(row)
	file.close()

	return header,rows


def get_omics():
	header, rows = open_csv("omics.csv")
	shape = len(rows), len(rows[0])-1

	omics = np.zeros(shape,dtype=float)
	pacientes = []
	genes = header[1:]

	for p in range(shape[0]):
		for g in range(shape[1]):
			omics[p,g] = float(rows[p][g+1])
		pacientes.append(rows[p][0])
	
	return omics,np.array(pacientes),genes


def get_subtype():
	header,rows = open_csv("subtype.csv")

	result = {}
	for i in range(len(header)):
		index = header[i].replace(".","-")
		result[index] = rows[0][i]
	
	return result

def normalize(dataset):
	return (dataset - np.min(dataset))/np.ptp(dataset)

###############################################################################

# Leemos los datos genómicos y clinicos
omics,pacientes,genes = get_omics()
subtype = get_subtype()

# Eliminamos los pacientes de subtipo NA
delete = []
for i in range(pacientes.shape[0]):
    pat = pacientes[i]
    if subtype[pat] == "NA" or subtype[pat] == "Normal":
        del subtype[pat]
        delete.append(i)
omics = np.delete(omics,delete,axis=0)
pacientes = np.delete(pacientes,delete,axis=0)

# Asignamos un número a cada categoría
categories = list(set(subtype.values()))
categories.sort()
mapping = { x: i for i,x in enumerate(categories) }
for pat in subtype:
    subtype[pat] = mapping[subtype[pat]]


# Normalizamos a rango (0,1) y obtenemos los conjuntos de entrenamiento y test
omics = StandardScaler().fit_transform(omics)
x_train, x_test, p_train, p_test = train_test_split(omics, pacientes, test_size=0.20, random_state=42)
print(omics.shape)

y_train = [subtype[pat] for pat in p_train]
y_test = [subtype[pat] for pat in p_test]
y_train = to_categorical(y_train) # one-hot-encoding
y_test = to_categorical(y_test)   # one-hot-encoding

###############################################################################

def Baseline(classes):
	model = Sequential( [
		Dense(1000, activation="tanh"),
		Dense(50, activation="tanh"),
		Dense(classes, activation="sigmoid")
	] )
 
	metrics = [
		tf.keras.metrics.AUC(),
		tfa.metrics.CohenKappa(classes),
		tfa.metrics.F1Score(classes)
 	]
	model.compile(loss = 'mse', metrics = metrics, optimizer = 'adam')
	return model

################################################################################

def SelectorAutoEncoder(n):
	def decoder(x):
		#x = Dense(512, activation="tanh")(x)
		x = Dense(x_train.shape[1], activation="sigmoid")(x)
		return x

	selector = ConcreteAutoencoderFeatureSelector(K = n, output_function = decoder, num_epochs = 1)
	selector.fit(x_train, x_train, x_test, x_test)
	indices = selector.get_support(indices = True)
	return x_train[:,indices], x_test[:,indices]

def PCA(n):
	pca = PCA(n)
	pca.fit(x_train)
	return pca.transform(x_train), pca.transform(x_test)

################################################################################

def KNN(x,y):
	param_grid = dict( n_neighbors = list(range(1, 11)))
	grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10, scoring='accuracy', return_train_score=False,verbose=1)
	grid_search = grid.fit(x,y)
	best_n = grid_search.best_params_["n_neighbors"]
	print(best_n)

	knn = KNeighborsClassifier(n_neighbors=best_n)
	knn.fit(x, y)
	return knn

def RF(x,y):
	n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
	criterion = ['gini','entropy']
	max_features = [None,'sqrt','log2',0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
	max_depth = [int(x) for x in np.linspace(20, 200, num = 10)]
	min_samples_split = [2, 5, 10]
	min_samples_leaf = [1, 2, 4]
	bootstrap = [True, False]
	class_weight = ['balanced','balanced_subsample',None]
	
	param_grid = {  'n_estimators': n_estimators,
					'max_features': max_features,
					'max_depth': max_depth,
					'min_samples_split': min_samples_split,
					'min_samples_leaf': min_samples_leaf,
					'bootstrap': bootstrap,
					'class_weight': class_weight
	}
	
	rf_random = RandomizedSearchCV(estimator = RandomForestClassifier(), param_distributions = param_grid, cv = 10)
	rf_random.fit(x, y)
	print(rf_random.best_params_)
 
	rf = RandomForestClassifier(
		  n_estimators = rf_random.best_params_["n_estimators"],
		  max_features = rf_random.best_params_["max_features"],
		  max_depth = rf_random.best_params_["max_depth"],
		  min_samples_split = rf_random.best_params_["min_samples_split"],
		  min_samples_leaf = rf_random.best_params_["min_samples_leaf"],
		  bootstrap = rf_random.best_params_["bootstrap"],
		  class_weight = rf_random.best_params_["class_weight"],
	)
	rf.fit(x,y)
 
	return rf

################################################################################

def Matriz(modelo,x_test,y_true):
	y_pred = modelo.predict(x_test)
	return tf.math.confusion_matrix(y_true.argmax(axis=1),y_pred.argmax(axis=1))

def Metrica(modelo,metrica,x_test,y_true):
	y_pred = modelo.predict(x_test)
	metrica.update_state(y_true, y_pred)
	return metrica.result().numpy()

################################################################################

model = Baseline(y_train.shape[1])

history = model.fit(
	x_train,
	y_train,
	epochs = 20,
	batch_size = 16,
	validation_data = (x_test, y_test)
)

x_train_AE,x_test_AE = SelectorAutoEncoder(128)
knn_ae = KNN(x_train_AE,y_train)
rf_ae = RF(x_train_AE,y_train)

x_train_PCA,x_test_PCA = PCA(128)
knn_pca = KNN(x_train_PCA,y_train)
rf_pca = RF(x_train_PCA,y_train)

################################################################################

metrics = {
    "auc": tf.keras.metrics.AUC(),
    "kappa": tfa.metrics.CohenKappa(len(mapping)),
    "f1": tfa.metrics.F1Score(len(mapping))
}

models = {
    "baseline": (model, x_test),
    "knn autoencoder": (knn_ae, x_test_AE),
    "knn pca": (knn_pca, x_test_PCA),
    "rf autoencoder": (rf_ae, x_test_AE),
    "rf pca": (rf_pca, x_test_PCA)
}

for name in models:
    mod, test = models[name]
    print(name)
    print(Matriz(mod,test,y_test))
    for n in metrics:
        mtr = metrics[n]
        print(n + " " + str(Metrica(mod,mtr,test,y_test)))
    print(" ")

print("\nClasses")
print(mapping)