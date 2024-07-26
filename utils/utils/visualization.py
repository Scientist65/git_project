import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_correlations(df):
    plt.figure(figsize=(16, 12))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.show()

def plot_pairwise_relationships(df):
    sns.pairplot(df)
    plt.show()
