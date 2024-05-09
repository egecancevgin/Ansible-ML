import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from ansible.executor.playbook_executor import PlaybookExecutor
from ansible.inventory.manager import InventoryManager
from ansible.parsing.dataloader import DataLoader
from ansible.vars.manager import VariableManager
from ansible.playbook.play import Play
import yaml
import ansible_runner



def ml_training(data_path):
    """ Trains an ML model to make predictions about automation."""
    df_encoded = pd.read_csv(data_path)
    #df_encoded = pd.get_dummies(df, columns=['Action'])
    label_encoder = LabelEncoder()
    df_encoded['Action'] = label_encoder.fit_transform(df_encoded['Action'])
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df_encoded['Message'])
    y = df_encoded.drop(columns=['Message'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Eğitimi
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Test seti doğruluğu:", accuracy)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    print("Precision:", precision)
    print("Recall:", recall)
    return clf, vectorizer, label_encoder


def ml_install_pb():
    """ Bazı ML modullerinin kurulumu yapan Ansible Playbook'unu (ml_install) çalıştırır."""
    print("ML kütüphaneleri yükleniyorr.")
    #runner = ansible_runner.Runner(config=None)
    playbook_path = "ml_install.yml"
    #results = runner.run(playbook_file=playbook_path)
    r = ansible_runner.run(playbook='/root/workspace/ml_install.yml')





def main():
    """ Driver function. """
    model, vectorizer, label_encoder = ml_training("task.csv")
    pr1 = "Scikit-learn modulü güncellenmeli."
    pr2 = "Nginx zaten mevcut."
    pr3 = "Node.js günceldir."
    pr4 = "Sistem güncellemeleri yapılmalı."
    pr5 = "Elasticsearch halihazırda mevcut"
    pr6 = "Docker yüklenmeli."
    pr7 = "ML modülleri yüklenmeli"
    pred = model.predict(vectorizer.transform([pr7]))
    predicted_class = label_encoder.inverse_transform(pred)[0]
    print(predicted_class)
    if predicted_class == "ml_install_pb": ml_install_pb()
    else: ml_install_pb()


if __name__ == '__main__':
    main()
