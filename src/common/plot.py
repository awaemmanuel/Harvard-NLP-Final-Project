
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


def pca_plots(vecs, df, vec_type, category_list):
    '''
    e.g.
    pca_plots(cv_vecs, df_train, "CountVectorizer" )
    '''
    pca_articles = PCA(n_components=2)
    principalComponents_articles = pca_articles.fit_transform(vecs)

    principal_articles_Df = pd.DataFrame(data = principalComponents_articles
                 , columns = ['principal component 1', 'principal component 2'])

    plt.figure()
    plt.figure(figsize=(10,10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component - 1',fontsize=20)
    plt.ylabel('Principal Component - 2',fontsize=20)
    plt.title(f"Principal Component Analysis of Articles dataset with {vec_type} vecs",fontsize=20)
    colors = cm.rainbow(np.linspace(0, 1, len(category_list)))
    for target, color in zip(category_list ,colors):
        indicesToKeep = df['category'] == target
        plt.scatter(principal_articles_Df.loc[indicesToKeep, 'principal component 1']
                   , principal_articles_Df.loc[indicesToKeep, 'principal component 2'], c = color.reshape(1,-1), s = 50)

    plt.legend(category_list,prop={'size': 15})