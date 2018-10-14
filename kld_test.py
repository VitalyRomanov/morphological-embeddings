import numpy as np

m1 = np.array([[1.,2.]])
m2 = np.array([[-3.,4.]])
log_v1 = np.array([[-.2, 1.]])
log_v2 = np.array([[.2, .4]])

xx = np.arange(-10,10,.1)
yy = np.arange(-10,10,.1)

def diag_inv(sigm):
    return np.diag(np.diag(sigm)**(-1.))

def mvg(v, m, log_s):

    sig = np.diag(np.squeeze(np.exp(log_s)))
    x = (v - m).reshape(-1,1)

    return np.squeeze(np.linalg.det(2*np.pi*sig)**(-.5) * np.exp(-.5*np.matmul(np.matmul(x.transpose(),diag_inv(sig)),x)))

kld = 0
for x in xx:
    for y in yy:
        point = np.array([[x,y]])
        # print(point)
        p1 = mvg(point, m1, log_v1) + 1e-20
        p2 = mvg(point, m2, log_v2) + 1e-20
        kld += p1 * np.log( p1 / p2 )

def kld_closed(m1, m2, log_v1, log_v2):
    det = np.linalg.det
    sig1 = np.diag(np.exp(np.squeeze(log_v1)))
    sig2 = np.diag(np.squeeze(np.exp(log_v2)))
    # print(np.log(det(sig2)/det(sig1)))
    # print(np.sum(log_v2 - log_v1))
    # print(np.trace(np.matmul(diag_inv(sig2), sig1)))
    # print(np.sum(np.exp(log_v1 - log_v2)))
    # print(np.matmul(np.matmul((m2 - m1) , diag_inv(sig2)), (m2 - m1).transpose()))
    # print(np.sum(np.square(m2-m1)*np.exp(-log_v2)))
    return .5 * (np.log(det(sig2)/det(sig1)) + \
                np.trace(np.matmul(diag_inv(sig2), sig1)) + \
                np.matmul(np.matmul((m2 - m1) , diag_inv(sig2)), (m2 - m1).transpose()))

def kld_closed_f(m1, m2, log_v1, log_v2):
    det = np.linalg.det
    sig1 = np.diag(np.exp(np.squeeze(log_v1)))
    sig2 = np.diag(np.squeeze(np.exp(log_v2)))
    return np.sum(log_v2 - log_v1) + \
                np.sum(np.exp(log_v1 - log_v2)) + \
                np.sum(np.square(m2-m1)*np.exp(-log_v2))

print(kld)
print(kld_closed(m1,m2,log_v1,log_v2))
print(kld_closed_f(m1,m2,log_v1,log_v2))
