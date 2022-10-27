from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from keras.utils import np_utils
import matplotlib 

hidden_features=[]

for i in range(200):
    hidden_features.append(preds)

hidden_features = np.array(hidden_features).mean(axis=0)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(hidden_features)
print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))

tsne = TSNE(n_components=2, verbose=1)
tsne_results = tsne.fit_transform(pca_result)

matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

plt.rcParams.update({'font.size': 25})
%matplotlib inline
Name=' '
fig = plt.figure(figsize=[10, 10])
color_map = np.argmax(y_test, axis=1)

colors = itertools.cycle(['darkgreen', 'red'])

for cl in range(2):
    indices = np.where(color_map==cl)
    indices = indices[0]
    plt.title(Name, fontsize=22)
    plt.ylabel('Dim_2', fontsize=22)
    plt.xlabel('Dim_1', fontsize=22)
    matplotlib.rc('xtick', labelsize=22) 
    matplotlib.rc('ytick', labelsize=22) 
    plt.scatter(tsne_results[indices,0], tsne_results[indices, 1], label=classes[cl], color=next(colors))

plt.rcParams.update({'font.size': 22})

plt.legend()

plt.show()

fig.savefig('{}.png'.format(Name), bbox_inches='tight')