import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

results = pd.read_csv('./runs/classify/train6/results.csv')

plt.figure(figsize = (8, 5))

sns.lineplot(data = results, x = 'epoch', y = 'train/loss', label = 'Train Loss', linewidth = 2, color = 'blue')
sns.lineplot(data = results, x = 'epoch', y = 'val/loss', label = 'Validation Loss', linewidth = 2, color = 'red')
plt.title('Loss vs Epochs', fontsize = 14)
plt.xlabel('Epochs', fontsize = 12)
plt.ylabel('Loss', fontsize = 12)
plt.legend()
plt.savefig('Project/results/Loss_vs_Epochs.jpg', dpi = 300)

plt.figure(figsize = (8, 5))
sns.lineplot(data = results, x = 'epoch', y = 'metrics/accuracy_top1', linewidth = 2, color = 'green')
plt.title('Validation Accuracy vs Epochs', fontsize = 14)
plt.xlabel('Epochs', fontsize = 12)
plt.ylabel('Accuracy (%)', fontsize = 12)
plt.savefig('Project/results/Validation_Accuracy_vs_Epochs.jpg', dpi = 300)