from matplotlib import pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import RandomForestRegressor

# Set up the dataset.
#  We'll create our dataset by drawing samples from Gaussians.

random_state = np.random.RandomState(seed=1)
n= 1000
p = 0.95

#X = random_state.normal(0, 1, n)
X = np.linspace(-5, 15, n)

e1 = random_state.normal(-10, 1, n)
e2 = random_state.normal(10, 1, n)
g = random_state.binomial(1, p, n)

e = [e1[i] if g[i] else e2[i] for i in range(n)]

cost = [1 if g[i] else 30 for i in range(n)]

Y = -2*X+e
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.hist(e)
ax2 = fig.add_subplot(212)
ax2.scatter(X, Y)


#f = Ridge(alpha=100)
f = LinearRegression()
#f = SVR(kernel='poly', C=1, gamma='auto', degree=2, epsilon=.1, coef0=1)
#f = MLPRegressor(random_state=0, activation='relu', hidden_layer_sizes=(16,8))
#f = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
           # max_features='auto', max_leaf_nodes=None,
           # min_impurity_decrease=0.0, min_impurity_split=None,
           # min_samples_leaf=1, min_samples_split=2,
           # min_weight_fraction_leaf=0.1, n_estimators=300, n_jobs=None,
           # oob_score=False, random_state=0, verbose=0, warm_start=False)
model = f.fit(X.reshape(-1, 1), Y.reshape(-1, 1), sample_weight=cost)
Yp = model.predict(X.reshape(-1, 1))

ax2.scatter(X, Yp)

plt.show()

