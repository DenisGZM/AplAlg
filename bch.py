import gf
import numpy as np
import random
import time

# t < (n-1)/2 
# n = 2 ** L - 1
def build_m(s,n):
    A =  np.zeros((n,n),int)
    b = np.zeros((n,1),int)
    for i in range(n):
        b[i] = s[n+i]
        for j in range(n):
            A[i,j] = s[i+j]
    return A, b

class BCH:
    def __init__(self, n, t):
        self.n = n
        self.t = t
        prim = gf.find_prim(n+1)
        self.pm = gf.gen_pow_matrix(prim)   
        self.R = self.pm[0:2*t,1]
        self.g = gf.minpoly(self.R,self.pm)[0]
        self.m = gf.deg(self.g) 
        self.k = self.n - self.m # word lenght

# Описание параметров:
# • U – набор исходных сообщений для кодирования, numpy.array-матрица,
#  бинарная матрица размера <число_сообщений>×k;
# Функция осуществляет систематическое кодирование циклического кода и 
# возвращает numpy.arrayматрицу с закодированными сообщениями размера <число_сообщений>×(k + m).
    def encode(self, U):
        if self.k != U.shape[1]:
            print("Wrong matrix dimension")
            return np.nan
        u = np.concatenate((U,np.zeros((U.shape[0],self.m),int)),axis = 1)
        v = np.zeros((u.shape[0],self.m),int)
        for i in range(u.shape[0]):
            v[i] = gf.expand(gf.polydiv(u[i],self.g,self.pm)[1],self.m)
        V = np.concatenate((U,v),axis = 1)
        return V

    def decode(self, w, method = 'euclid'):
        f = open(method + '.txt', 'a')
        u = np.zeros((w.shape[0],w.shape[1]),int)
        for i in range(w.shape[0]):
            s = gf.polyval(w[i],self.R,self.pm)
            if np.count_nonzero(s) == 0:
                u[i] = w[i]
                continue
            if method == 'pgz':
                t = time.clock()
                A, b = build_m(s,self.t)
                L = gf.linsolve(A,b,self.pm)
                step = self.t - 1
                while type(L) == type(np.nan) and step > 0:
                    A, b = build_m(s,step)
                    L = gf.linsolve(A,b,self.pm)
                    step -= 1 
                if type(L) == type(np.nan):
                    u[i] = -1
                    continue
                L = np.concatenate((L,[1]))  
                t = time.clock() - t
            if method == 'euclid':
                t = time.clock()
                s = np.concatenate((s[::-1],[1]))
                z = np.zeros(2*self.t + 2,int)
                z[0] = 1
                L = gf.euclid(z,s,self.pm,self.t)[2]
                t = time.clock() - t
            f.write(str(t) + '\n')
            val = gf.polyval(L,self.pm[:,1].reshape((self.pm.shape[0])),self.pm)
            pos = []
            for a in range(val.size):
                if val[a] == 0:
                    pos.append(a)
            if len(pos) != L.size-1:
                u[i] = -1
                continue
            u[i] = w[i]
            for j in pos:
                u[i,j] = (u[i,j] + 1) % 2
        f.close()
        return u

    def dist(self):
        edge = 2 ** self.k
        min = self.k
        v = np.zeros((edge-1,self.k),int)
        for i in range(1,edge):
            u = bin(i)[2:]
            for j in range(1, len(u)*2 - 1, 2):
                u = u[:j] + ' ' + u[j:]
            tmp = np.fromstring(u, dtype = int, sep = ' ')
            v[i-1] = gf.expand(tmp,self.k)
        u = self.encode(v)
        for i in range(u.shape[0]):
            if min > np.count_nonzero(u[i]):
                min = np.count_nonzero(u[i])
        return min
            
def graph(t):
    n = 2*t + 1
    n = (n ^ int(bin(n)[3:],2)) << 1
    step = 1
    while step < 3 and n < 127:
        a = BCH(n-1,t)
        k = a.dist()
        if k > 2*t+1:
            print(n-1, " " , t)
        n = n << 1
        step += 1


def graphics(k):
    for i in range(1,k):
        graph(i)

def testing(times, n = 0, t = 0, met = 'euclid'):
    if n == 0:
        q = random.randint(3,7)
        n = 2 ** q - 1
        
    if t == 0:
        t = random.randint(1, (n-1)//2 )
        
    a = BCH(n,t)
    success = 0
    mismatch = 0
    abandon = 0
    t_count = []
    for _ in range(times):
        word = np.array([[random.randint(0,1) for _ in range(a.k)]])
        error_count = 0
        # print("Word\n",word)
        right_code = a.encode(word)
        skywed_code = right_code.copy()
        for i in range(a.k):
            if random.random() < t/n:
                error_count += 1
                skywed_code[0,i] = (skywed_code[0,i] + 1) % 2
        if(error_count > t):
            t_count.append(error_count)
        if met == 'both':
            decoded1 = a.decode(skywed_code,'pgz')
            decoded2 = a.decode(skywed_code,'euclid')
            if np.count_nonzero(decoded1 - decoded2) == 0:
                decoded = decoded1
            else:
                mismatch += 1
                continue
        else:
            decoded = a.decode(skywed_code, met)
        if np.count_nonzero(decoded + 1) == 0:
            abandon += 1
            continue
        if np.count_nonzero(decoded - right_code) == 0:
            success += 1
            continue
        else:
            mismatch += 1
        # print("Right code\n",right_code)
        # print("Skywed code\n",skywed_code)
        # print("Decoded code\n", decoded)

    print('Code len = ',n)
    print('Errors allowed = ',t)
    print("Word len = ",a.k)
    print("Success = ", success/times * 100, '%')
    print("Abandon = ", abandon/times * 100, '%')
    print("Mismatch = ", mismatch/times * 100, '%')
    print("Count when error count was > than required: ", len(t_count), t_count)


testing(100,met = 'euclid')

