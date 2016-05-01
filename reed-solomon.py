import numpy as np
n = 17
k = 15
d = 3
# alpha = [3, 9, 10, 13, 5, 15, 11, 16, 14, 8, 7, 4, 12, 2, 6, 1, 0]
alpha = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
alpha_inv = [0, 1, 9, 6, 13, 7, 3, 5, 15, 2, 12, 14, 10, 4, 11, 8, 16]


V = np.matrix(np.ones((n, k)))
for i in range(n):
	for j in range(k):
		V[i, j] = (alpha[i] ** j) % 17


Vande = V[:k, :] # k * k

U_inv = np.matrix(np.zeros((k, k)))
L_inv = np.matrix(np.zeros((k, k)))


def L_inv_denominator(i, j):
	r = 1
	for p in range(0, i+1):
		if p == j:
			continue
		r *= (alpha[j] - alpha[p])
	return r

for i in range(k):
	for j in range(k):
		if i < j:
			L_inv[i, j] = 0
		elif i == 0 and j == 0:
			L_inv[i, j] = 1
		else:
			denominator = L_inv_denominator(i, j) % 17
			L_inv[i, j] = alpha_inv[denominator]

for i in range(k):
	for j in range(k):
		if i == j:
			U_inv[i, j] = 1
		elif j == 0:
			U_inv[i, j] = 0
		else:
			if i == 0:
				U_inv[i, j] = round(0 - U_inv[i, j - 1] * alpha[j - 1]) % 17
			else:
				U_inv[i, j] = round(U_inv[i-1, j-1] - U_inv[i, j-1] * alpha[j - 1]) % 17


# print L_inv

Vande_inv = U_inv * L_inv
row, col = Vande_inv.shape
for i in range(row):
	for j in range(col):
		Vande_inv[i, j] %= 17

def encode(message):
	b = np.array(message)
	b = np.asmatrix(b).T
	codeword = V * b
	row, col = codeword.shape
	for i in range(row):
		for j in range(col):
			codeword[i, j] = codeword[i, j] % 17
	return np.array(codeword).reshape(-1,).tolist()


def decode(codeword):
	b = np.array(codeword[:k])
	b = np.asmatrix(b).T

	solution = Vande_inv * b
	row, col = solution.shape
	for i in range(row):
		for j in range(col):
			solution[i, j] = round(solution[i, j] % 17)

	return np.array(solution).reshape(-1,).tolist()


codeword = encode([5, 1, 3, 2, 4, 10, 3, 14, 12, 9, 0, 8, 16, 7, 15])
print "codeword:\t\t", codeword

message = decode(codeword)
print "message:\t\t", message

codeword2 = encode(message)
print "encode again:\t", codeword2
