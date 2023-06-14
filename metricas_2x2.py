# Dataset:
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


y_true = [0, 0, 0, 0, 1, 1]  # ground truth
y_pred = [0, 1, 0, 0, 0, 1]  # model.predict()
# y_pred = [0, 1, 0, 0, 1, 1] # model.predict()

# Matriz de confusión:
cm = confusion_matrix(y_true, y_pred)
print("confusion matrix: \n", cm)
# gráfica cm
plt.figure(figsize=(8, 4))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Prediction', fontsize=12)
plt.ylabel('Real', fontsize=12)
plt.show()

# Exactitud:
acc = accuracy_score(y_true, y_pred)
print("accuracy: ", acc)

# Sensibilidad:
recall = recall_score(y_true, y_pred)
print("recall: ", recall)

# Precisión:
precision = precision_score(y_true, y_pred)
print("precision: ", precision)

# Especificidad
# 'specificity' is just a special case of 'recall'.
# specificity is the recall of the negative class
specificity = recall_score(y_true, y_pred, pos_label=0)
print("specificity: ", specificity)

# Puntuación F1:
f1 = f1_score(y_true, y_pred)
print("f1 score: ", f1)

# Área bajo la curva:
auc = roc_auc_score(y_true, y_pred)
print("auc: ", auc)

# Curva ROC
plt.figure()
lw = 2
plt.plot(roc_curve(y_true, y_pred)[0], roc_curve(y_true, y_pred)[1],
         color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# R Score (R^2 coefficient of determination)
R = r2_score(y_true, y_pred)
print("R2: ", R)
