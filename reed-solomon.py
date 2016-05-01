import numpy as np
n = 17
k = 15
d = 3
# alpha = [3, 9, 10, 13, 5, 15, 11, 16, 14, 8, 7, 4, 12, 2, 6, 1, 0]
alpha = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
alpha_inv = [0, 1, 9, 6, 13, 7, 3, 5, 15, 2, 12, 14, 10, 4, 11, 8, 16]
F_17_inv = {0: np.nan, 1: 1, 2: 9, 3: 6, 4: 13, 5: 7, 6: 3, 7: 5, 8: 15,
			9: 2, 10: 12, 11: 14, 12: 10, 13: 4, 14: 11, 15: 8, 16: 16}


V = np.matrix(np.ones((n, k)))
for i in range(n):
	for j in range(k):
		V[i, j] = (alpha[i] ** j) % 17


def L_inv_denominator(i, j, X):
	r = 1
	for p in range(0, i+1):
		if p == j:
			continue
		r *= (X[j] - X[p])
	return r


def compute_Vande_inv(Vande):
	"""
	compute the inverse of Vandermonde matrix
	:param Vande: m * m dimension matrix
	:return: m * m dimension matrix
	"""
	X = []
	row, col = Vande.shape
	for i in range(row):
		X.append(Vande[i, 1])
	U_inv = np.matrix(np.zeros((row, row)))
	L_inv = np.matrix(np.zeros((row, row)))

	for i in range(row):
		for j in range(row):
			if i < j:
				L_inv[i, j] = 0
			elif i == 0 and j == 0:
				L_inv[i, j] = 1
			else:
				denominator = L_inv_denominator(i, j, X) % 17
				L_inv[i, j] = F_17_inv[denominator]

	for i in range(row):
		for j in range(row):
			if i == j:
				U_inv[i, j] = 1
			elif j == 0:
				U_inv[i, j] = 0
			else:
				if i == 0:
					U_inv[i, j] = round(0 - U_inv[i, j - 1] * X[j - 1]) % 17
				else:
					U_inv[i, j] = round(U_inv[i - 1, j - 1] - U_inv[i, j - 1] * X[j - 1]) % 17

	Vande_inv = U_inv * L_inv
	for i in range(row):
		for j in range(row):
			Vande_inv[i, j] %= 17
	return Vande_inv


def encode(message):
	b = np.array(message)
	b = np.asmatrix(b).T
	codeword = V * b
	row, col = codeword.shape
	for i in range(row):
		for j in range(col):
			codeword[i, j] %= 17
	return np.array(codeword).reshape(-1,).tolist()


def decode(codeword):
	b = np.matrix(np.ones((k, 1)))
	pos = []
	count = 0
	index = 0
	while count < k:
		if codeword[index] != -1:
			b[count, 0] = codeword[index]
			pos.append(index)
			count += 1
		index += 1

	Vande = np.matrix(np.ones((k, k)))
	for i in range(k):
		for j in range(k):
			Vande[i, j] = (alpha[pos[i]] ** j) % 17
	Vande_inv = compute_Vande_inv(Vande)

	solution = Vande_inv * b
	row, col = solution.shape
	for i in range(row):
		for j in range(col):
			solution[i, j] = round(solution[i, j] % 17)

	return np.array(solution).reshape(-1,).tolist()


print "encoding message"
codeword1 = encode([5, 1, 3, 2, 4, 10, 3, 14, 12, 9, 0, 8, 16, 7, 15])
print "codeword1:\t\t", codeword1

print "decoding message"
message1 = decode(codeword1)
print "message1:\t\t", message1

print "decoding from 1 erasure"
codeword2 = encode([12, 13, 3, 1, 2, 0, 9, 14, 3, 2, 0, 6, 7, 9, 11])
codeword2[3] = -1		# -1 denote missing code '?'
print "codeword2:\t\t", codeword2
message2 = decode(codeword2)
print "message2:\t\t", message2

print "decoding from 2 erasure"
codeword3 = encode([9, 13, 6, 1, 2, 0, 12, 8, 5, 2, 0, 6, 7, 1, 11])
codeword3[2] = -1
codeword3[13] = -1
message3 = decode(codeword3)
print "message3:\t\t", message3


