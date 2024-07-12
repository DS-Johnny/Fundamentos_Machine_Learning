# IMPORT LIBRARIES
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics as mt


# Load Dataset
df = pd.read_csv('train.csv')

# Features
features = ['idade', 'saldo_atual', 'divida_atual', 'renda_anual', 'valor_em_investimentos', 'taxa_utilizacao_credito', 'num_emprestimos', 'num_contas_bancarias', 'num_cartoes_credito', 'dias_atraso_dt_venc',
     'num_pgtos_atrasados', 'num_consultas_credito', 'taxa_juros']
x_train = df[features].values
# Label
y_train = df.loc[:,'limite_adicional'].values


neighbors = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
for k in neighbors:
    # DEFINING TRAINMENT PARAMETERS
    knn_classifier = KNeighborsClassifier( n_neighbors=k )

    # TRAINING ALGORITHM
    knn_classifier.fit( x_train, y_train)

    df_result = df.copy()
    y_pred = knn_classifier.predict( x_train )
    df_result['classificacao'] = y_pred

    # scikit.metrics CONFUSION MATRIX
    conf_matrix = mt.confusion_matrix(y_train, y_pred)
    

    # Accuracy
    accuracy = mt.accuracy_score(y_train, y_pred)

    # Precision
    pres_con = mt.precision_score( y_train, y_pred, average="binary", pos_label='Conceder' )
    pres_neg = mt.precision_score( y_train, y_pred, average="binary", pos_label='Negar' )

    # Recall
    recall_con = mt.recall_score( y_train, y_pred, average='binary', pos_label='Conceder' )
    recall_neg = mt.recall_score( y_train, y_pred, average='binary', pos_label='Negar' )

    print(30*'-=')
    print(f'KNeighbors: {k}')
    print(f'Confusion Matrix: \n{conf_matrix}\nAccuracy: {accuracy*100:.2f}%\n')
    print(f'Precision:\nConceder: {pres_con*100:.2f}%\nNegar: {pres_neg*100:.2f}%\n')
    print(f'Recall:\nConceder: {recall_con*100:.2f}%\nNegar: {recall_neg*100:.2f}%\n')

