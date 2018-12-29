import numpy as np

# expand with zeros to size = size  
def expand(p, size):
    for _ in range(size-p.size):
        p = np.insert(p,0,0)
    return p
    
def decrease(p):
    while p.size > 1 and p[0] == 0:
        p = np.delete(p,0)
    return p

#Generate correspondance table
def gen_pow_matrix(primpoly):
    b_pr = bin(primpoly)[2:]
    deg_max = len(b_pr)
    equ = int(b_pr[1:],2)
    dp = 2 ** (deg_max-1) - 1
    pm = np.zeros((dp, 2), dtype = int)
    alpha = 1
    for i in range(dp):
        alpha = alpha << 1
        if len(bin(alpha)[2:]) == deg_max:
            alpha ^= equ
            alpha = int(bin(alpha)[3:],2)
        pm[i,1] = alpha
        pm[alpha-1,0] = i + 1
    return pm



def add(X, Y):
    return X ^ Y

def sum(X, axis = 0):
    if axis == 0:
        tmp = np.zeros((1, X.shape[1]), int)
        for i in range(X.shape[0]):
            tmp = np.bitwise_xor(X[i,:],tmp)
    elif axis == 1:
        tmp = np.zeros(X.shape[0], int)
        for i in range(X.shape[1]):
            tmp = np.bitwise_xor(X[:,i], tmp)
        tmp = tmp.reshape((X.shape[0], 1))
    else:
        print("Wrong axis!")
        return np.nan
    return tmp

def prod(X, Y, pm):
    tmp = np.zeros(X.shape,int)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i,j] == 0 or Y[i,j] == 0:
                tmp[i,j] = 0
                continue
            x_in = pm[X[i,j] - 1, 0]
            y_in = pm[Y[i,j] - 1, 0]
            tmp[i,j] = pm[(x_in + y_in - 1) % pm.shape[0],1]
    return tmp

def divide(X, Y, pm):
    tmp = np.zeros(X.shape,int)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i,j] == 0:
                tmp[i,j] == 0
                continue
            if Y[i,j] == 0:
                print("Division by zero!")
                return np.nan
            x_in = pm[X[i,j]-1, 0]
            y_in = pm[Y[i,j]-1, 0]
            tmp[i,j] = pm[(x_in - y_in - 1) % pm.shape[0],1]
    return tmp

# Описание параметров:
# • A – квадратная матрица из элементов поля Fq2;
# • b – вектор из элементов поля Fq2;
# • pm – матрица соответствия между десятичным и степенным представлением в поле Fq2;
# Функция возвращает решение СЛАУ в случае невырожденности A и numpy.nan иначе.
def linsolve(A, b, pm):
    A = A.copy()
    b = b.copy()
    for j in range(A.shape[0]):
        step = j + 1
        while A[j,j] == 0 and step < A.shape[0]:
            A[[j,step]] = A[[step,j]]
            b[[j,step]] = b[[step,j]]
            step += 1
        div = A[j,j]    
        if div == 0:
            return np.nan
        tmp = np.zeros((1,A.shape[1]),int) + div
        A[j] = divide(A[j].reshape((1,A.shape[1])),tmp,pm)
        b[j] = divide(b[j].reshape((1,1)),np.array(div,int).reshape((1,1)),pm)
        for i in range(j+1, A.shape[0]):
            mul = A[i,j]
            tmp = np.zeros((1,A.shape[1]),int) + mul
            A[i] = add(A[i],prod(A[j].reshape((1,A.shape[1])),tmp,pm))
            b[i] = add(b[i].reshape((1,1)),prod(b[j].reshape((1,1)),np.array(mul,int).reshape((1,1)),pm))
    x = np.zeros(A.shape[0],int)
    for i in range(A.shape[0]-1,-1,-1):
        for j in range(i,A.shape[0]):
            b[i] = add(b[i].reshape((1,1)),prod(A[i,j].reshape((1,1)),x[j].reshape((1,1)),pm))
        x[i] = b[i]
    return x


def minpoly(x, pm):
    in_x = []
    cur_ind_x = -1
    for i in range(x.size):
        cur_ind_x = pm[x[i]-1, 0] - 1
        if not(cur_ind_x in in_x):
            in_x.append(cur_ind_x)
            adj_in = (cur_ind_x*2 + 1) % pm.shape[0]
            while adj_in != cur_ind_x:
                in_x.append(adj_in)
                adj_in = (adj_in*2 + 1) % pm.shape[0]
    res = [1]
    for i in in_x:
        tmp = [i]
        for j in range(1,len(res)):
            if res[j] != 0:
                tmp.append((res[j] + i) % pm.shape[0])
            else:
                tmp.append(-1)
        res.append(0)
        for j in range(1,len(res)):
            if tmp[j-1] == -1:
                continue
            if res[j] == 0:
                res[j] = tmp[j-1] + 1
                continue
            loc = pm[res[j] - 1, 1] ^ pm[tmp[j -1], 1]
            if loc == 0:
                res[j] = 0
                continue
            cur_ind_x = pm[loc-1, 0]
            res[j] = cur_ind_x
    for i in range(1,len(res)):
        if res[i] == pm.shape[0]:
            res[i] = 1
    return np.array(res), np.sort(pm[in_x,1])   

# Описание параметров:
# • p – полином из Fq2[x], numpy.array-вектор коэффициентов, начиная со старшей степени;
# • x – вектор из элементов поля Fq2;
# • pm – матрица соответствия между десятичным и степенным представлением в поле Fq2;
# Функция возвращает значения полинома p для набора элементов x.
def polyval(p, x, pm):
    res = []
    p = decrease(p)
    for i in range(x.size):
        res.append(0)
        # a * (val)^(p.size-1) + ...
        for j in range(p.size):
            if p[j] == 0:
                continue
            cur_in = (pm[p[j]-1, 0] - 1 + (pm[x[i]-1, 0]) * (p.size - 1 - j)) % pm.shape[0]
            res[i] ^= pm[cur_in,1]
    return np.array(res,int)

def polyprod(p1, p2, pm):
    p1 = decrease(p1)
    p2 = decrease(p2)
    if np.count_nonzero(p1) == 0 or np.count_nonzero(p2) == 0:
        return np.zeros(1, dtype = int)
    res = np.zeros(p1.size + p2.size - 1, int)
    for i in range(p1.size):
        if p1[i] == 0:
            continue
        for j in range(p2.size):
            if p2[j] == 0:
                continue
            in_1 = pm[p1[i] - 1, 0]
            in_2 = pm[p2[j] - 1, 0]
            if res[i+j] == 0:
                res[i+j] = ((in_1 + in_2 - 1) % pm.shape[0]) + 1
                continue
            loc = pm[res[i+j]-1, 1] ^ pm[(in_1 + in_2 - 1) % pm.shape[0], 1]
            if loc == 0:
                res[i+j] = 0
                continue
            res[i+j] = pm[loc - 1, 0]
    for i in range(0, len(res)):
        if res[i] in range(1, pm.shape[0]+1):
            res[i] = pm[res[i]-1, 1]
    return res
      


# Описание параметров:
# • p1, p2 – полиномы из Fq2[x], numpy.array-вектор коэффициентов, начиная со старшей степени;
# • pm – матрица соответствия между десятичным и степенным представлением в поле Fq2;
# Функция осуществляет деление с остатком многочлена p1 на многочлен p2. Функция возвращает
# кортеж из переменных:
# • частное, numpy-array-вектор коэффициентов, начиная со старшей степени;
# • остаток от деления, numpy-array-вектор коэффициентов, начиная со старшей степени.
def polydiv(p1, p2, pm):
    p1 = decrease(p1)
    p2 = decrease(p2)
    if np.count_nonzero(p1) == 0:
        return np.zeros(2, dtype = int)
    if np.count_nonzero(p2) == 0:
        print("Division by zero")
        return np.nan
    q = np.zeros(max(0,p1.size-p2.size) + 1, int)
    div = p1.copy()
    iter = 0
    while div.size >= p2.size and div[0] != 0:
        cur_in = (pm[div[0] - 1, 0] + pm.shape[0] - pm[p2[0] - 1, 0] - 1) % pm.shape[0]
        q[iter] = pm[cur_in,1]
        div[0] = 0
        for i in range(1,p2.size):
            if p2[i] == 0:
                continue
            if div[i] == 0:
                div[i] =  pm[((pm[p2[i] - 1, 0] + cur_in) % pm.shape[0]), 1]
                continue
            div[i] =  div[i] ^ pm[((pm[p2[i] - 1, 0] + cur_in) % pm.shape[0]), 1] 
        div = decrease(div)
        iter += 1 
    return np.array(q, int), div


#size - count of zeros in the front
def deg(p):
    if np.count_nonzero(p) == 0:
        return -1
    pos = 0
    while pos < p.size and p[pos] == 0:
        pos += 1
    return p.size - pos - 1

# Описание параметров:
# • p1, p2 – полиномы из Fq2[x], numpy.array-вектор коэффициентов, начиная со старшей степени;
# • pm – матрица соответствия между десятичным и степенным представлением в поле Fq2;
# • max_deg – максимально допустимая степень остатка, число, если равно нулю, то алгоритм Евклида
# работает до конца;
# Функция реализует расширенный алгоритм Евклида для пары многочленов p1 и p2. Функция возвращает кортеж из переменных:
# • остаток, numpy-array-вектор коэффициентов, начиная со старшей степени;
# • коэффициент при p1, numpy-array-вектор коэффициентов, начиная со старшей степени;
# • коэффициент при p2, numpy-array-вектор коэффициентов, начиная со старшей степени.
def euclid(p1, p2, pm, max_deg=0):
    if np.count_nonzero(p1) < 0:
        return
    g0 = np.zeros(1, dtype = int)
    g1 = np.zeros(1, dtype = int)
    f0 = np.zeros(1, dtype = int)
    f1 = np.zeros(1, dtype = int)
    if p1.size >= p2.size:
        p0, p1 = p1, p2
    else:
        p0, p1 = p2, p1
    g0[0] = 1
    f1[0] = 1
    while deg(p0) > max_deg and np.count_nonzero(p1) != 0:
        q, r = polydiv(p0,p1,pm)
        r = polyprod(p1, q, pm)
        p0, p1 = p1, expand(p0, r.size) ^ expand(r, p0.size)
        r = polyprod(g1, q, pm)
        g0, g1 = g1, expand(g0, r.size) ^ expand(r, g0.size)
        r = polyprod(f1, q, pm)
        f0, f1 = f1, expand(f0, r.size) ^ expand(r, f0.size)
        p1 = decrease(p1)
        f0 = decrease(f0)
        f1 = decrease(f1)
        g0 = decrease(g0)
        g1 = decrease(g1)
    return p0, g0, f0

def find_prim(n):
    f = open("primpoly.txt", 'r')
    g = f.read().split(', ')
    f.close()
    for a in g:
        if int(a) - n < n and int(a) - n > 0:
            return int(a)
    
    