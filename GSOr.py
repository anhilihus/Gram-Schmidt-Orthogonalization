import numpy as np
import matplotlib.pyplot as plt

m = int(input("Enter the number of rows of the matrix: "))
n = int(input("Enter the number of columns of the matrix: "))
A = np.empty((m, n))

for j in range(n):
    print(f"Enter the elements of signal {j+1}:")
    for i in range(m):
        A[i, j] = float(input(f"Enter the element at index {i}: "))

def gram_schmidt(A):
    Q = np.empty_like(A)
    Q[:, 0] = A[:, 0] / np.linalg.norm(A[:, 0])
    R = np.zeros((n, n))
    for j in range(1, n):
        proj = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            proj -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(proj)
        Q[:, j] = proj / R[j, j]
    return Q, R

Q, R = gram_schmidt(A)
C = np.dot(Q.T, A)

print("The matrix of coefficients of the orthogonal decomposition of A is:")
print(C)
print("The orthogonal basis obtained through Gram-Schmidt orthogonalization is:")
print(Q)
print("To verify the columns of Q are orthonormal:")
print(np.dot(Q.T, Q))

symbols = C.T
plt.scatter(symbols.real, symbols.imag)
plt.xlabel('In-Phase')
plt.ylabel('Quadrature')
plt.title('Constellation Diagram')
plt.grid()
plt.show()
