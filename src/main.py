import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn.neural_network import MLPClassifier
import random
from sklearn.model_selection import KFold
import pickle
from time import time

def main():
    p = 0.15
    print("Ejecutando Main")
    data = pd.read_csv( "../data/verbal_autopsies_clean.csv",header=0, skiprows=lambda i: i > 0 and random.random() > p)
    print('Se ha leído correctamente el archivo CSV')
    classes = np.array(data['gs_text34'])
    indices = np.array(data['newid'])

    data_tfidf = create_tfidf(data)

    # ponemos la clase al final
    instances = np.column_stack((data_tfidf, classes))
    n_iterations = 5
    # se utilizará una muestra con el 63.2% de las instancias para el train dataset
    n_size = int(len(instances) * 1.00)
    print('prueba1')
    instances1 = np.column_stack((indices, instances))
    # bootstrap_method(instances1, n_iterations, n_size)

    print('======================================================')
    print('Calculando el k-fold para Multilayer Perceptron (MLP)')
    print(k_fold(10, instances))


def create_tfidf(data_frame):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data_frame['open_response'].values.astype('U'))
    print('Se ha creado correctamente el TFIDF')
    return tfidf_matrix.A


# zero rule algorithm for classification
def zero_rule_algorithm_classification(train, test):
    # devuelve la clase cuyo valor es más común en el train dataset
    # obtenemos la lista con las clases que aparecen en train dataset
    output_values = [row[-1] for row in train]
    # obtenemos la que aparece un mayor número de veces
    prediction = max(set(output_values), key=output_values.count)

    predicted = [prediction for i in range(len(test))]

    print(predicted)

    return predicted

def parallel_shuffle(a, b):
    # https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    if len(a) == len(b):
        p = np.random.permutation(len(a))
        return a[p], b[p]

def mlp_classifier(test, train):
    # hidden_layer_size-> utilizamos una capa oculta y la hacemos variar
    # variaciones de 5 en 5 comenzando en 5 terminando en 100
    print('===========================================')
    print('Calculando Modelo Multilayer Perceptron ')
    print('===========================================')
    print('Variaciones de la capa oculta de 5 en 5, hasta llegar a 100')
    layer_size = 50
    best_score = 0
    best_model = None
    best_train = None
    best_test = None
    while layer_size <= 100:
        model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(layer_size, ), random_state=None)
        # ajuste del modelo
        model.fit(train[:, :-1], train[:, -1])
        # evaluación del modelo
        predictions = model.predict(test[:, :-1])
        # score representa la acertividad de predicción del modelo
        score = accuracy_score(test[:, -1], predictions)
        print(score)
        if score > best_score:
            best_score = score
            best_model = model
            best_train = train
            best_test = test
        layer_size += 5
    return best_score, best_model, best_train, best_test

def graph_mlp_classifier_model(train,test):
    print('Generando gráfico MLP')
    stats = list()
    layer_size = 30
    while layer_size <= 100:
        model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(layer_size,), random_state=None)
        # ajuste del modelo
        model.fit(train[:, :-1], train[:, -1])
        # evaluación del modelo
        predictions = model.predict(test[:, :-1])
        # score representa la acertividad de predicción del modelo
        score = accuracy_score(test[:, -1], predictions)
        stats.append(score)
        layer_size += 5
    pyplot.bar(np.arange(50., 100, 5), stats, width=5)




def decision_tree_clasifier(test, train):
    model = DecisionTreeClassifier()
    # ajuste del modelo
    model.fit(train[:, :-1], train[:, -1])
    # evaluación del modelo
    predictions = model.predict(test[:, :-1])
    # score representa la acertividad de predicción del modelo
    return accuracy_score(test[:, -1], predictions)

def decision_tree_clasifier_model(train, n_size):
    model = DecisionTreeClassifier()
    model.fit(train[:, :-1], train[:, -1])
    return model



def k_fold(k, instances):
    start_time = time()
    print('===========================================')
    print('Calculando K_Fold Cross Validation para MLP')
    print('===========================================')
    print('Se utilizará para k=10')
    X = instances
    # y = classes
    kf = KFold(n_splits=k, shuffle=True)
    best_score = 0
    best_model = None
    best_train = None
    best_test = None
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        score, model, train, test = mlp_classifier(X_test, X_train)
        if score > best_score:
            best_score = score
            best_model = model
            best_train = train
            best_test = test
        print(best_score)

    print('El mejor score de acierto del MLP bajo K_Fold CV es: '+ str(best_score))
    print('Sacando el resultado del modelo variando el hidden layer o capa oculta')
    elapsed_time = time() - start_time
    print('Tiempo de ejecución del método de k-fold CV con MLP' + str(elapsed_time))
    graph_mlp_classifier_model(best_train, best_test)
    return best_score

def bootstrap_method(instances, n_iterations, n_size):
    start_time = time()
    stats = list()
    model=None
    # representa el score máximo que se ha alcanzado con el train model
    max_value=0
    print('=========================================================')
    print('Calculando varios boostraps para Decision Tree Clasifier')
    print('=========================================================')
    print('Se realizará una prueba a '+ str(n_iterations) + ' ireaciones')
    print('Por cada iteración se calculará el score ')
    for i in range(n_iterations):
        # preparando  train and test sets
        train = resample(instances, n_samples=n_size)
        indices = train[:, 0]
        test = np.array([x for x in instances if x[0] not in indices])
        # testeamos con el modelo de decision tree clasifier
        score = decision_tree_clasifier(test[:, 1:], train[:, 1:])
        if score > max_value:
            max_value = score
            model=decision_tree_clasifier_model(test, train)
        print(score)
        stats.append(score)
    # plot scores
    pyplot.hist(stats)
    pyplot.show()
    # instervalos de confianza
    alpha = 0.95
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))
    print('Valor máximo del training ' + str(max_value))
    print('Para obtener un %.1f%% de acertividad el intervalo de confianza será entre: %.1f%% y %.1f%%' % (alpha * 100, lower * 100, upper * 100))
    print('Este intervalo es el porcentaje que nos clasificará bien con el clasificador Decision Tree ')
    print('Se ha generado el mejor modelo para dichas condiciones')
    elapsed_time = time() - start_time
    print('Tiempo de ejecución del método de bootstraping con Decision Tree Classifier' + str(elapsed_time))

def save_results(self, indiv_results, save_path):
    with open(save_path, 'w') as file:
        file.write(self.results_to_text(indiv_results))

def save_model(model, save_path):
    # https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence/4529901#4529901
    with open(save_path, 'wb') as output:  # Overwrites any existing file.
         pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

def results_to_text(self, indiv_results):
    text_results = {}
    num_results = len(indiv_results)
    class_names = np.unique(self._classes)
    num_classes = len(class_names)
    cum_precision, cum_recall, cum_accuracy, cum_f_score, cum_kappa, \
    cum_tpr, cum_fnr, cum_fpr, cum_tnr = 0, 0, 0, 0, 0, 0, 0, 0, 0
    for class_name in class_names:
        class_cum_precision, class_cum_recall, class_cum_accuracy, class_cum_f_score, class_cum_kappa, \
        class_cum_tpr, class_cum_fnr, class_cum_fpr, class_cum_tnr = 0, 0, 0, 0, 0, 0, 0, 0, 0

        for results in indiv_results:
            class_cum_precision += results.precision(class_name)
            class_cum_recall += results.recall(class_name)
            class_cum_accuracy += results.accuracy(class_name)
            class_cum_f_score += results.f_score(class_name)
            class_cum_kappa += results.kappa(class_name)
            class_cum_tpr += results.tpr(class_name)
            class_cum_fnr += results.fnr(class_name)
            class_cum_fpr += results.fpr(class_name)
            class_cum_tnr += results.tnr(class_name)

        class_avg_precision = class_cum_precision / num_results
        class_avg_recall = class_cum_recall / num_results
        class_avg_accuracy = class_cum_accuracy / num_results
        class_avg_f_score = class_cum_f_score / num_results
        class_avg_kappa = class_cum_kappa / num_results
        class_avg_tpr = class_cum_tpr / num_results
        class_avg_fnr = class_cum_fnr / num_results
        class_avg_fpr = class_cum_fpr / num_results
        class_avg_tnr = class_cum_tnr / num_results

        cum_precision += class_avg_precision
        cum_recall += class_avg_recall
        cum_accuracy += class_avg_accuracy
        cum_f_score += class_avg_f_score
        cum_kappa += class_avg_kappa
        cum_tpr += class_avg_tpr
        cum_fnr += class_avg_fnr
        cum_fpr += class_avg_fpr
        cum_tnr += class_avg_tnr

        tmp_text = '=========================\n'
        tmp_text += 'CLASE: {}\n'.format(class_name.upper())
        tmp_text += '=========================\n'
        tmp_text += 'Valores medios:\n'
        tmp_text += 'Prec.(%): \t{}\n'.format(class_avg_precision)
        tmp_text += 'Recall: \t{}\n'.format(class_avg_recall)
        tmp_text += 'Accuracy: \t{}\n'.format(class_avg_accuracy)
        tmp_text += 'F-Score: \t{}\n'.format(class_avg_f_score)
        tmp_text += 'Kappa: \t\t{}\n'.format(class_avg_kappa)
        tmp_text += 'TPR: \t\t{}\n'.format(class_avg_tpr)
        tmp_text += 'FNR: \t\t{}\n'.format(class_avg_fnr)
        tmp_text += 'FPR: \t\t{}\n'.format(class_avg_fpr)
        tmp_text += 'TNR: \t\t{}\n'.format(class_avg_tnr)

        text_results[class_name] = tmp_text

    avg_precision = cum_precision / num_classes
    avg_recall = cum_recall / num_classes
    avg_accuracy = cum_accuracy / num_classes
    avg_f_score = cum_f_score / num_classes
    avg_kappa = cum_kappa / num_classes
    avg_tpr = cum_tpr / num_classes
    avg_fnr = cum_fnr / num_classes
    avg_fpr = cum_fpr / num_classes
    avg_tnr = cum_tnr / num_classes

    avg_text = '=========================\n'
    avg_text += 'MEDIA DE TODAS LAS CLASES\n'
    avg_text += '=========================\n'
    avg_text += 'Prec.(%): \t{}\n'.format(avg_precision)
    avg_text += 'Recall: \t{}\n'.format(avg_recall)
    avg_text += 'Accuracy: \t{}\n'.format(avg_accuracy)
    avg_text += 'F-Score: \t{}\n'.format(avg_f_score)
    avg_text += 'Kappa: \t\t{}\n'.format(avg_kappa)
    avg_text += 'TPR: \t\t{}\n'.format(avg_tpr)
    avg_text += 'FNR: \t\t{}\n'.format(avg_fnr)
    avg_text += 'FPR: \t\t{}\n'.format(avg_fpr)
    avg_text += 'TNR: \t\t{}\n'.format(avg_tnr)

    final_text = avg_text
    for class_name in text_results:
        final_text += '\n{}'.format(text_results[class_name])

    return final_text
def results_to_txt():
    nm=None

if __name__ == '__main__':
    main()
