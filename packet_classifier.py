# full_packet_classifier.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import export_text
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 60)
    print("СИСТЕМА КЛАССИФИКАЦИИ ПРИОРИТЕТОВ СЕТЕВЫХ ПАКЕТОВ")
    print("=" * 60)
    
    # Шаг 1-2: Генерация данных
    print("\n[1] Генерация синтетических данных...")
    np.random.seed(42)
    n_samples = 1000
    
    def generate_packet_data(n_samples):
        packet_size = np.random.choice([64, 128, 256, 512, 1024, 1500], n_samples)
        protocol = np.random.choice([0, 1, 2], n_samples)
        frequency = np.random.randint(1, 1000, n_samples)
        latency = np.random.randint(1, 500, n_samples)
        
        return pd.DataFrame({
            'packet_size': packet_size,
            'protocol': protocol,
            'frequency': frequency,
            'latency': latency
        })
    
    data = generate_packet_data(n_samples)
    
    def assign_priority(row):
        if (row['protocol'] == 2 and row['packet_size'] <= 128 and row['latency'] < 50):
            return 3
        elif (row['protocol'] == 1 and row['packet_size'] <= 256 and row['latency'] < 100):
            return 2
        elif (row['protocol'] == 0 and row['packet_size'] <= 512 and row['frequency'] < 500):
            return 1
        else:
            return 0
    
    data['priority'] = data.apply(assign_priority, axis=1)
    print(f"   Сгенерировано {n_samples} записей")
    print(f"   Распределение классов: {data['priority'].value_counts().to_dict()}")
    
    # Шаг 3: Разделение данных
    print("\n[2] Разделение на обучающую и тестовую выборки...")
    X = data[['packet_size', 'protocol', 'frequency', 'latency']]
    y = data['priority']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"   Обучающая выборка: {len(X_train)} образцов")
    print(f"   Тестовая выборка: {len(X_test)} образцов")
    
    # Шаг 4: Обучение модели
    print("\n[3] Обучение модели дерева решений...")
    dt_classifier = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    dt_classifier.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, dt_classifier.predict(X_train))
    print(f"   Точность на обучающей выборке: {train_acc:.4f}")
    
    # Шаг 5: Тестирование
    print("\n[4] Тестирование модели...")
    y_pred = dt_classifier.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"   Точность на тестовой выборке: {test_acc:.4f}")
    
    print("\n   Матрица ошибок:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Шаг 6: Тестирование на различных типах трафика
    print("\n[5] Тестирование на специфических типах трафика...")
    test_cases = pd.DataFrame({
        'packet_size': [64, 1500, 256, 128, 512, 1024, 64, 256],
        'protocol': [2, 0, 1, 2, 0, 1, 0, 1],
        'frequency': [100, 10, 300, 50, 200, 5, 800, 400],
        'latency': [20, 200, 50, 10, 150, 300, 25, 80],
        'description': [
            'ICMP Ping', 'Large TCP', 'VoIP', 'ICMP Control',
            'Web Traffic', 'File Download', 'TCP SYN Flood', 'UDP Streaming'
        ]
    })
    
    test_predictions = dt_classifier.predict(
        test_cases[['packet_size', 'protocol', 'frequency', 'latency']]
    )
    test_cases['predicted'] = test_predictions
    print(test_cases[['description', 'predicted']].to_string(index=False))
    
    # Шаг 7: Визуализация
    print("\n[6] Визуализация дерева решений...")
    plt.figure(figsize=(20, 10))
    plot_tree(dt_classifier, 
              feature_names=['packet_size', 'protocol', 'frequency', 'latency'],
              class_names=['Low(0)', 'Medium(1)', 'High(2)', 'Critical(3)'],
              filled=True, rounded=True, fontsize=10)
    plt.title('Дерево решений для классификации приоритетов пакетов')
    plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 60)
    print("РАБОТА ЗАВЕРШЕНА УСПЕШНО!")
    print("=" * 60)

if __name__ == "__main__":
    main()