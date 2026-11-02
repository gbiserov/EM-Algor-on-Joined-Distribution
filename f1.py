import numpy as np
import scipy as sp
import mixem as mm
import random

theta = 1 
p1 = 1/2
p2 = 1/4 * theta 
p3 = 1/4 * (1-theta)
p4 = 1/4 * (1-theta)
p5 = 1/4 * theta

prob = [p1,p2,p3,p4,p5]

x = random.randint(1,100)
print(x)

samples = [random.random() for _ in range(10)]
print(samples)

from scipy.stats import multinomial

counts = [10,5,0,0,5]
n=sum(counts)
p = multinomial.pmf(counts, n=n, p=prob)
print("multinomial p:", p)

### em algorithm

np.random.seed(42)

data1 = np.concatenate([
np.random.normal(0, 1, 500),
np.random.normal(5, 1, 500)
])

distr = [
    mm.distribution.NormalDistribution(mu= -1, sigma =2),
    mm.distribution.NormalDistribution(mu= 4,sigma=2)
]

weights,distr,log_likelihood = mm.em(
    data1,
    distr,
    initial_weights=[0.5,0.5],
    max_iterations = 200
)

print("Estimated mixture weights:", weights)
print("Estimated means:", [d.mu for d in distr])
print("Estimated standard deviations:", [d.sigma for d in distr])
print("Log-likelihood:", log_likelihood)


