import numpy as np 

# Archivo .py solicitado 

# Funciones pedidas por el TP 

# Funciones pedidas por el TP  ----------- Ejercicio 1

#  ---> No requiere código explícito 

# Funciones pedidas por el TP  ----------- Ejercicio 2 ------------- Item a) --------------

def obtenerQ3d(imPts, objtPts):
    """
    Construcción de la matriz Q en base a la definición del marco teórico, 
    recibe dos arrays, imPts y objtPts. Retorna la matriz Q de dimensiones 2N x 12.    
    """

    U, V, W = objtPts[:,0], objtPts[:,1], objtPts[:,2]

    Q = np.zeros((2*objtPts.shape[0],12))

    for j in range(objtPts.shape[0]):

        u = imPts[2*j]
        v = imPts[2*j+1]

        Q[2*j, :] =     [U[j], V[j], W[j], 1,   0,    0,    0,   0, -u*U[j], -u*V[j], -u*W[j], -u]
        Q[2*j + 1, :] = [ 0,    0,    0,   0,  U[j], V[j], W[j], 1, -v*U[j], -v*V[j], -v*W[j], -v]
        

    return Q

# Funciones pedidas por el TP  ----------- Ejercicio 2 ------------- Item b) --------------

def resolverDLT3D(Q):
    '''
    Parámetros que recibe: la matriz de calibración Q de 2nx12.
    Output esperado: la matriz M que es igual a K[R t].
    '''
    #Dado que en el mundo real no existe un vector m distinto de cero en el núcleo de Q, entonces por la demostración dada en el punto (1b) vemos que podemos obtener M tomando la última columna de la matriz V, siendo V la matriz dada en la desomposición SVD de la matriz Q del input.
    hatU, hatSig, hatV = svd_reducida(Q,'max',1e-15) 
    M = hatV[:,-1] 
    M = np.reshape(M,(3,4)) #Como el resultado se obtiene de considerar a M vectorizado, una vez obtenido aplicamos un reshape para darle la forma matricial original correspondiente.
    return M

# Funciones pedidas por el TP  ----------- Ejercicio 2 ------------- Item c) e Item d) --------------

def devolverMatricesCalibracion3d(M,op=(False,False)):
    '''
    Parámetros que recibe: M, la matriz dada por la múltplicación de las variables extrínsecas e instrínsecas de la cámara; y op, una tupla de booleanos (de forma predeterminada con dos False) que al indicarle True en alguna o ambas de sus posiciones impone que el sesgo sea nulo y/o la igualdad entre los alpha_u y alpha_v.
    Output esperado: R,t,K las variables extrínsecas e intrínsecas de la cámara.
    '''
    #Dado que queremos hallar K, R y t tal que M = K[R t] pensamos al lado derecho de la igualdad como una multplicación por bloques obteniendo que M = [KR | Kt] = [A | b]. Con esta última reesctritura (A = KR y b = Kt) podemos despejar y hallar cada matriz deseada.
    A = M[:,:3]
    b = M[:,3]

    #Definimos una matriz que permute las filas y columnas (dependiendo de por cual lado se multiplique) y que nos será útil para, con cálculos algebráicos, obtener el output deseado.
    P = np.array([[0,0,1],
                  [0,1,0],
                  [1,0,0]])

    At = transpuesta(A)
    PAt = arroba(P,At)
    PAtP = arroba(PAt,P)

    q,r = QR_con_HH(PAtP)

    rt = transpuesta(r)
    Prt = arroba(P,rt)
    K = arroba(Prt,P) 

    #K = K/K[2][2]

    qt = transpuesta(q)
    Pqt = arroba(P,qt)
    R = arroba(Pqt,P)

    for s in range(3):
        if K[s][s] < 0:
            K[:,s] = (-1)*K[:,s]
            R[s,:] = (-1)*R[s,:]

    Kinv = inversa(K)
    t = arroba(Kinv,b)

    #K = K/K[2][2]

    if op[0] == True:
        K[0][0] = K[1][1]
    if op[1] == True:
        K[0][1] = 0

    return R,t,K


# Funciones pedidas por el TP  ----------- Ejercicio 3

# Funciones pedidas por el TP  ----------- Ejercicio 3 ------------- Punto 2 --------------

def obtenerQ2d(imgPts, objPts):
    """
    Construcción de la matriz Q para el cálculo de homografía H
    Recibe un arreglo 1D imgPts y un arreglo 2D objPts 
    y retorna la matriz Q preasignada de dimensiones 2N x 9.
    """
    U, V = objPts[:, 0], objPts[:, 1]
    N = objPts.shape[0]
    
    Q = np.zeros((2 * N, 9))
    
    for j in range(N):
        u = imgPts[2*j]
        v = imgPts[2*j+1]
        
        Q[2*j, :]   = [U[j], V[j], 1,   0,    0,  0,  -u*U[j], -u*V[j], -u]
        Q[2*j+1, :] = [ 0,    0,   0, U[j], V[j], 1,  -v*U[j], -v*V[j], -v]
        
    return Q

# Funciones pedidas por el TP  ----------- Ejercicio 3 ------------- Punto 3 --------------

def armarA(Hlst):
    n = len(Hlst)
    A = np.zeros((2 * n, 6))
    
    for i, H in enumerate(Hlst):
        c1, c2 = H[:, 0], H[:, 1]
        
        A[2*i, :] = [c1[0]*c2[0], c1[0]*c2[1] + c1[1]*c2[0], c1[1]*c2[1], c1[2]*c2[0] + c1[0]*c2[2], c1[2]*c2[1] + c1[1]*c2[2], c1[2]*c2[2]]

        A[2*i + 1, :] = [c1[0]*c1[0] - c2[0]*c2[0],
                         c1[0]*c1[1] + c1[1]*c1[0] - c2[0]*c2[1] - c2[1]*c2[0],
                         c1[1]*c1[1] - c2[1]*c2[1],
                         c1[2]*c1[0] + c1[0]*c1[2] - c2[2]*c2[0] - c2[0]*c2[2],
                         c1[2]*c1[1] + c1[1]*c1[2] - c2[2]*c2[1] - c2[1]*c2[2],
                         c1[2]*c1[2] - c2[2]*c2[2]]
                         
    return A

# Funciones pedidas por el TP  ----------- Ejercicio 3 ------------- Punto 4 --------------

def calcularMatrizK2D(A):
    '''
    Calcula K a partir de A (2N x 6) por el método de Zhang.
    Resuelve min ||A b|| con ||b|| = 1, arma B simétrica, elige el signo para que B
    sea definida positiva, factoriza B = L L^T (Cholesky) y devuelve K = (L^T)^{-1} con K[2,2] = 1.
    '''
    # min ||A b||: autovector de A^T A con el menor autovalor (sin descartar ninguno)
    AtA = arroba(transpuesta(A), A)
    AtA = AtA / np.max(np.abs(AtA)) 
    S, D = diagRH(AtA, 1e-12, K=1000)
    i_min = 0
    for i in range(6):
        if D[i, i] < D[i_min, i_min]:
            i_min = i

    b = S[:, i_min]

    B = np.array([[b[0], b[1], b[3]],
                  [b[1], b[2], b[4]],
                  [b[3], b[4], b[5]]])
    if not esSDP(B):
        B = (-1) * B
    if not esSDP(B):
        raise ValueError('B no es definida positiva: no se puede aplicar Cholesky.')

    L = calculaCholesky(B)     
    K = inversa(transpuesta(L))
    return K / K[2, 2]

# Funciones pedidas por el TP  ----------- Ejercicio 3 ------------- Punto 5 --------------

def obtenerPose2D(Hj, K):
    Kinv = inversa(K)
    C1,C2,C3 = Hj[:,0],Hj[:,1],Hj[:,2]
    KinvC1,KinvC2,KinvC3 = np.zeros(Kinv.shape[0]),np.zeros(Kinv.shape[0]),np.zeros(Kinv.shape[0])
    
    KinvC1 = arroba(Kinv, C1)
    KinvC2 = arroba(Kinv, C2)
    KinvC3 = arroba(Kinv, C3)

    λ = 1/norma(KinvC1,2)
    r1 = λ*KinvC1
    r2 = λ*KinvC2
    r3 = np.array([r1[1]*r2[2]-r1[2]*r2[1],-r1[0]*r2[2]+r1[2]*r2[0],r1[0]*r2[1]-r1[1]*r2[0]])
    Rj = np.zeros((3,3))
    Rj[:,0], Rj[:,1], Rj[:,2] = r1,r2,r3
    tj = λ*KinvC3
    
    return Rj,tj

# Funciones pedidas por el TP  ----------- Ejercicio 4

# ACÁ NO NOS PIDEN NINGUNA FUNCIÓN EXPLÍCITAMENTE 

# Funciones pedidas por el TP  ----------- Ejercicio 5

# ACÁ NO NOS PIDEN NINGUNA FUNCIÓN EXPLÍCITAMENTE 

# Funciones pedidas por el TP  ----------- Ejercicio 6

# Funciones pedidas por el TP  ----------- Ejercicio 6 ------------- item a) --------------

#  ---> No requiere código explícito 

# Funciones pedidas por el TP  ----------- Ejercicio 6 ------------- item b) --------------

def calculoLambdaMu(I0, J0):
    """
    Recibe los vectores I0 y J0
    Devuelve los escalares lambda y mu
    """
    # Norma al cuadrado 
    norma_I0 = np.sum(I0 * I0)
    norma_J0 = np.sum(J0 * J0)

    producto_interno_I0_J0 = np.sum(I0 * J0)

    # numero complejo C, Lamnda * u = -I0 * J0 ? 
    Re = norma_I0 - norma_J0
    Im = -2.0 * producto_interno_I0_J0

    # Meditar modulo_de_C
    modulo_de_C = np.sqrt(Re * Re + Im * Im)

    lam = np.sqrt((modulo_de_C + Re) / 2.0)

    # Im solo nos inetresa para estimar el signo 
    if Im < 0:
        signo = -1.0
    else:
        signo = 1.0

    mu = signo * np.sqrt((modulo_de_C - Re) / 2.0)
    
    return lam, mu

# Funciones pedidas por el TP  ----------- Ejercicio 6 ------------- item c) --------------

def calcularSalidaPOSIT(Is, Js, fx, mO, MO):
    """
    Recibe los vectores Is y Js, la distancia focal dada fx, el píxel m0 y el punto real M0 
    Devuelve los ejes (i, j, k), la profundidad del punto de referencia Z0 y la pose (R, t)
    """

    # Calcado del marco teórico
    norma_Is = norma(Is, 2)
    norma_Js = norma(Js, 2)
    Z0 = fx / norma_Is

    i = Is / norma_Is
    j = Js / norma_Js

    k = np.array([
        i[1]*j[2] - i[2]*j[1],
        i[2]*j[0] - i[0]*j[2],
        i[0]*j[1] - i[1]*j[0]
    ])

    R = np.zeros((3, 3))
    R[0, :] = i
    R[1, :] = j
    R[2, :] = k

    x0 = mO[0]
    y0 = mO[1]

    X0 = (Z0 / fx) * x0
    Y0 = (Z0 / fx) * y0

    OM0 = np.array([X0, Y0, Z0])
        
    t = OM0 - arroba(R, MO)
        
    return i, j, k, Z0, R, t

# Funciones pedidas por el TP  ----------- Ejercicio 6 ------------- item d) --------------

def evaluarPose(R, t, obj_vecs, img_vecs, K):
    """
    Recibe la pose (R, t), los puntos 3D (obj_vecs), los píxeles 2D (img_vecs) y la matriz K dada
    Devuelve la cantidad de proyecciones válidas (Z > 0) y el RMS 
    """
    n = obj_vecs.shape[0]
    validos = 0
    suma_error_sq = 0.0

    t = np.reshape(t, -1)
    for i in range(n):

        # Coordenadas de obajeto a cámara a través de la ec 1 
        P_obj = obj_vecs[i]
        P_cam = arroba(R, P_obj) + t

        # La purga que dicta Selecciónde soluciones y factibilidad de la pg 20
        if P_cam[2] > 0:
            validos += 1

        p_img = arroba(K, P_cam)


        # (u, v) proyectado
        u_proy = p_img[0] / p_img[2]
        v_proy = p_img[1] / p_img[2]

        # (u, v) sacado de la foto 
        u_real = img_vecs[i, 0]
        v_real = img_vecs[i, 1] 


        # para rms completo, el arma dada por la pg 21 
        error = (u_proy - u_real)**2 + (v_proy - v_real)**2
        suma_error_sq += error

    rms = np.sqrt(suma_error_sq/n)

    return validos, rms 

# Funciones pedidas por el TP  ----------- Ejercicio 6 ------------- item e) --------------

def aplicarPOSIT(obj_vecs, img_vecs, K, MaxIt, atol):
    """
    Iteración POSIT
    """
    
    N = obj_vecs.shape[0]

    # Asumir el error inicial epsilon para cada P 
    eps = np.zeros(N)

    # ----------- acordate de quitar la columna label 
    MO = obj_vecs[0]
    a_vecs = obj_vecs - MO





    # calculo de A+, para luego calcular I0 = A+ b 
    hatU, hatSig, hatV = svd_reducida(a_vecs, k="max", tol=1e-15)

    # Sing^+
    r = len(hatSig)
    Sig_inv = np.zeros((r, r))
    for i in range(r):
        Sig_inv[i, i] = 1.0 / hatSig[i]


    # A^+ = V * Sing^+ * U^T         -------- dim = 3 x N 
    hatV_Sing = arroba(hatV, Sig_inv)
    A_pseudo_inv = arroba(hatV_Sing, transpuesta(hatU))





    # desmenuzando la info de K 
    fx = K[0,0]
    u0 = K[0,2]
    v0 = K[1,2]






    # centrar las coordenadas x e y con u0 y v0 

    x = img_vecs[:, 0] - u0
    y = img_vecs[:, 1] - v0 
    
    mO = np.array([x[0], y[0]])




    # vector normal al piso y unitario 
    vector_normal = np.array([0.0, 0.0, 1.0])


    mejor_R = np.eye(3) # Matriz
    mejor_t = np.zeros(3) # vector

    # arranca el bucle en el punto 1 de iteración POSIT

    for _ in range(MaxIt):

        # 1,) Construcción de b_u y b_v
        b_u = x * (1.0 + eps) - x[0]
        b_v = y * (1.0 + eps) - y[0]
 
        # 2.)resolver los sistemas I0 = A+ * b 
        I0 = arroba(A_pseudo_inv, b_u)
        J0 = arroba(A_pseudo_inv, b_v)

        # 3.) caso coplanar
        lam, mu = calculoLambdaMu(I0, J0)

        # lo casos posibles se dividen en solo 2 por el producto de signos, para que cuadre con la ec

        # Positivos
        I1 = I0 + lam * vector_normal
        J1 = J0 + mu * vector_normal
        # RAMA 2 (con -mu)
        I2 = I0 - lam * vector_normal
        J2 = J0 - mu * vector_normal

        # 4.) recuperar R y t y todo lo demás, para los dos casos 
        _, _, k1, Z01, R1, t1 = calcularSalidaPOSIT(I1, J1, fx, mO, MO)
        _, _, k2, Z02, R2, t2 = calcularSalidaPOSIT(I2, J2, fx, mO, MO)

        # y evaluar pose tmb

        validos1, rms1 = evaluarPose(R1, t1, obj_vecs, img_vecs, K)
        validos2, rms2 = evaluarPose(R2, t2, obj_vecs, img_vecs, K)

        # y buscamos las mejor pose tomando como primera prioridad la cantidad de Z validos 
        # si Z negativo es descartado en evaluar pose 

        if validos1 == N and validos2 == N:
            # Desempat por el menor error RMS
            if rms1 < rms2:
                mejor_R, mejor_t, mejor_eje_k, mejor_Z0 = R1, t1, k1, Z01
            else:
                mejor_R, mejor_t, mejor_eje_k, mejor_Z0 = R2, t2, k2, Z02
        elif validos1 == N:
            mejor_R, mejor_t, mejor_eje_k, mejor_Z0 = R1, t1, k1, Z01
        elif validos2 == N:
            mejor_R, mejor_t, mejor_eje_k, mejor_Z0 = R2, t2, k2, Z02
        else:
            # Si ninguna sirve compleatmente, menor error 
            if rms1 < rms2:
                mejor_R, mejor_t, mejor_eje_k, mejor_Z0 = R1, t1, k1, Z01
            else:
                mejor_R, mejor_t, mejor_eje_k, mejor_Z0 = R2, t2, k2, Z02

        #5} calcular el coso 5, no se como carajos se llama

        eps_actual = np.zeros(N)
        for i in range(N):

            # eps_i = (1 / Z0) * (a_i · k)  --> k vector direccional, el que sale del producto cruz i x j
            eps_actual[i] = (1.0 / mejor_Z0) * np.sum(a_vecs[i] * mejor_eje_k) 

        # 6) el máximo error en cada punto 
        diferencia = np.max(np.abs(eps_actual - eps))

        if diferencia < atol:
            print(f"la diferencia fue menor antes de terminar MaxIt, RMS_1 = {rms1}, RMS_2 = {rms2}")
            break

        eps = eps_actual
    return mejor_R, mejor_t

# Funciones pedidas por el TP  ----------- Ejercicio 6 ------------- item f) --------------

#  ---> No requiere código explícito 
















#===========================================================================================================================
























# Funciones auxiliares

# resolverDLT2D aux del ejercicio 4

def resolverDLT2D(A):
    '''
    Parámetros que recibe:
    Que retornará:
    (Recordar comentar el código y dar a las variables nombres explicativos)
    '''
    hatU, hatSig, hatV = svd_reducida(A)
    H = hatV[:,-1]
    H = np.reshape(H,(3,3))
    return H

# armar_RT aux del ejercicio 5

def armar_Rt(R, t):
    Rt = np.zeros((3, 4))
    Rt[:, :3] = R
    Rt[:, 3] = np.reshape(t, -1)
    return Rt

def arroba(A, B):
    A_Filas, A_Colum = np.shape(A)
    
    es_vector_1d = (len(np.shape(B)) == 1) #[n,]
    
    if es_vector_1d:
        B_filas = np.shape(B)[0]
        B_Colum = 1 
    else:
        B_filas, B_Colum = np.shape(B)
        
    if A_Colum != B_filas:
        print('Las dimensiones no cuadran')
        return
        
    if es_vector_1d:
        res = np.zeros(A_Filas, dtype=float)
        for i in range(A_Filas):
            suma = 0
            for k in range(A_Colum):
                suma += A[i, k] * B[k]  
            res[i] = suma
    else:
        res = np.zeros((A_Filas, B_Colum), dtype=float)
        for i in range(A_Filas):
            for j in range(B_Colum):
                suma = 0
                for k in range(A_Colum):
                    suma += A[i, k] * B[k, j]
                res[i, j] = suma
                
    return res

def transpuesta(A):

    if len(np.shape(A)) == 1:
        return A.reshape(1, -1)
        
    filas, colum = np.shape(A)
    res = np.zeros((colum, filas))
    
    for i in range(filas):
        for j in range(colum):
            res[j, i] = A[i, j]
    
    return res  

def funcTL(A, v, k):
    w = np.copy(v)

    for _ in range(k):

        w_prima = arroba(A, w) 
        norma_w = norma(w_prima, 2)

        if norma_w > 0:
            w = w_prima / norma_w
        else:
            w = np.zeros(np.shape(w_prima))

    return w

def transponer(lista_vectores, n):
    r = len(lista_vectores)
    
    if r == 0:
        return np.zeros((n, 0))
    
    matriz_T = np.zeros((n, r))

    for i in range(r):
        for j in range(n):
            matriz_T[j, i] = lista_vectores[i][j]
            
    return matriz_T

def determinante_3x3(A):
    return (A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]) -
            A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0]) +
            A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]))

def productoInterno(v1, v2):
    if len(v1) != len(v2):
        raise ValueError('Los vectores deben tener la misma dimensión.')
    resultado = 0
    for i in range(len(v1)):
        resultado += v1[i] * v2[i]
    return resultado

# Funciones aux del ejercicio 5

def tiempoDesdeNombre(nombre):

    partes = nombre.replace('.jpg', '').split('_')     # ['img', '10', '18', '47', '351']

    # Esto en base al tiempo hora - min - seg - mlseg del nombre de los archibos 

    return int(partes[1]) * 3600 + int(partes[2]) * 60 + int(partes[3]) + int(partes[4]) / 1000

def aCalle(pix, H_inversa):
    pos = np.zeros((pix.shape[0], 2))
    for i in range(pix.shape[0]):
        p = arroba(H_inversa, np.array([pix[i, 0], pix[i, 1], 1.0]))
        pos[i, 0] = p[0] / p[2]
        pos[i, 1] = p[1] / p[2]
    return pos

def velocidades(pos, t):
    v = np.zeros(pos.shape[0] - 1)
    for i in range(pos.shape[0] - 1):
        distancia = np.sqrt((pos[i+1, 0] - pos[i, 0])**2 + (pos[i+1, 1] - pos[i, 1])**2)
        v[i] = distancia / (t[i+1] - t[i])
    return v

def calcularVelocidades(pix, idx, H_inversa, tiempos):
    pos = aCalle(pix, H_inversa)
    t = tiempos[np.array(idx)]
    v = np.zeros(pos.shape[0] - 1)
    for i in range(pos.shape[0] - 1):
        distancia = np.sqrt((pos[i+1, 0] - pos[i, 0])**2 + (pos[i+1, 1] - pos[i, 1])**2)
        v[i] = distancia / (t[i+1] - t[i])                  # m/s
    distancia_total = np.sqrt((pos[-1, 0] - pos[0, 0])**2 + (pos[-1, 1] - pos[0, 1])**2)
    v_media = distancia_total / (t[-1] - t[0])
    return v, v_media




#===========================================================================================================================



















# Funciones del labo en orden cronológico



# Funciones del labo 1 --------------------------------


def error(x, y):
    x_f64 = float(x)
    y_f64 = float(y)
    res = abs(x_f64 - y_f64)
    return res 

def error_relativo(x, y):
    if x == 0 and y == 0:
        return 0
    elif x == 0:
        return float('inf')
    res = error(x, y) / abs(x)
    return res


def matricesIguales(A,B):
    mA = np.array(A)
    mB = np.array(B)

    if np.shape(mA) != np.shape(mB):
        return False
    
    #para reshape
    cantElem = 1
    for dim in np.shape(mA):
        cantElem *= dim

    vectorA = np.reshape(mA, cantElem)
    vectorB = np.reshape(mB, cantElem)

    for i in range(cantElem):
        if not np.isclose(vectorA[i], vectorB[i], atol=1e-08):
            return False

    return True 

# Funciones del labo 2 --------------------------------


def rota(theta):
    res = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    return res

def escala(s):
    vector = np.array(s)
    n = np.shape(vector)[0]

    res = np.eye(n) * vector

    return res

def rota_y_escala(theta,s):
    rotada = rota(theta)
    escalada = escala(s)
    res = escalada @ rotada
    return res 


def afin(theta, s, b):

    matriz = rota_y_escala(theta, s)
    vector = np.array(b)

    res = np.array([
        [matriz[0,0], matriz[0,1], vector[0]],
        [matriz[1,0], matriz[1,1], vector[1]],
        [0,0,1]
    ]) 

    return res 

def trans_afin(v,theta,s,b):
    vectorDim2 = np.array(v)
    vectorDim3 = np.array([vectorDim2[0], vectorDim2[1], 1.0])

    matrizAfin = afin(theta, s, b)

    wDim3 = matrizAfin @ vectorDim3

    wDim2 = wDim3[0:2]

    return wDim2


# Funciones del labo 3 --------------------------------

def norma(x, p):

    if p == "inf":
        return abs(max(x, key = abs))

    res = 0
    for xi in x:
        res += (abs(xi)**p)

    res = res**(1/p)

    return res 

def normaliza(X, p):
    Y = []
    for x in X:
        normaActual = norma(x,p)
        xArray = np.array(x)
        if normaActual > 0:
            Y.append(xArray / normaActual)
        else:
            Y.append(xArray)

    return Y


def normaMatMC(A, q, p, Np):

    # dim = (f, c) 
    n = np.shape(A)[1]

    vectores = [np.random.randn(n) for _ in range(Np)]

    # entrada ||A||p = 1 
    vectoresNormalizados = normaliza(vectores, p)

    normaMaxima = -1.0
    vectorMaximo = None

    for v in vectoresNormalizados:

        vectorModificado = A @ v 

        # Medición con q 
        normaActual = norma(vectorModificado, q)

        if normaMaxima < normaActual:
            normaMaxima = normaActual 
            vectorMaximo = vectorModificado

    return normaMaxima, vectorMaximo



def normaExacta(A, p=[1, 'inf']):
    Array = np.array(A)
    m, n = np.shape(Array)
    res = []

    if not isinstance(p, list):
        p = [p]

    for norma in p:
        if norma == 'inf':
            # filas
            normaInf = max(  sum(abs(Array[i, j]) for j in range(n)) for i in range(m)  ) 
            res.append(normaInf)
        elif norma == 1:
            # columnas
            normaUno = max(  sum(abs(Array[i, j]) for i in range(m)) for j in range(n)  ) 
            res.append(normaUno)

    return res 


def condMC(A, p):
    
    Np = 10000
    AInv = np.linalg.inv(A)

    normaA = normaMatMC(A, p, p, Np)[0]
    normaAInv = normaMatMC(AInv, p, p, Np)[0]

    return normaA * normaAInv

def condExacto(A, p):

    A_inv = np.linalg.inv(A)
    
    res = normaExacta(A, p) * normaExacta(A_inv, p)

    return res 



# Funciones del labo 4 --------------------------------


def calculaLU(A):

    # defensa
    if A is None:
        return None, None, 0
        
    A_arr = np.array(A, dtype=float)
    m = A_arr.shape[0]
    n = A_arr.shape[1]
    
    if m != n:
        return None, None, 0

    
    Ac = A_arr.copy()
    cant_op = 0

    
    for p in range(n):
        if Ac[p, p] == 0:
            return None, None, 0

        # plantado desde la columna se calcula el factor y se va bajando por esa misma columna desde el pivote 
        for i in range(p + 1, n):
            # factor de eliminación y creador de 0 
            factor = Ac[i, p] / Ac[p, p]

            # modifico directamente 
            Ac[i, p] = factor
            cant_op += 1  # una división

            # se planta esta vze en la fila y va multiplicando los valores por el factor desde el pivote 
            for j in range(p + 1, n):
                Ac[i, j] -= factor * Ac[p, j]
                cant_op += 2  # una multiplicación y una resta
    L = np.eye(n)
    U = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i > j:
                L[i, j] = Ac[i, j]
            else:
                U[i, j] = Ac[i, j]  

    return L, U, cant_op



def res_tri(L, b, inferior=True):

    L_arr = np.array(L) 
    b_arr = np.array(b) 
    n = np.shape(L_arr)[0]

    #defensa
    if np.shape(L_arr)[1] != n or np.shape(b_arr)[0] != n:
        return None

    
    x = np.zeros(n)
    
    if inferior:
        # patrón de iteraciones anotado en el cuaderno 
        for i in range(n):
            suma = 0.0 # suma acumulada para el despeje de x 
            for j in range(i):
                suma += L_arr[i, j] * x[j] 
            x[i] = (b_arr[i] - suma) / L_arr[i, i]
            
    else:
        # mismo patrón pero al revés 
        for i in range(n - 1, -1, -1):
            suma = 0.0
            for j in range(i + 1, n):
                suma += L_arr[i, j] * x[j]
            x[i] = (b_arr[i] - suma) / L_arr[i, i]
            
    return x



# busco A^-1 
def inversa(A):
    L, U, _ = calculaLU(A)

    if L is None:
        return None

    
    # A * A^-1 = I 
    # Si x es cada columna de A^-1 
    # Ax = I, y como A = LU 
    # LUx = I --> e cada columna de I 
    # LUx = e 

    n = np.shape(A)[0]

    A_inversa = np.zeros((n,n)) 
    iden = np.eye(n) 

    for i in range(n):
        e_i = iden[:,i] # cada columna de la identidad 

        # será la b de el sistema de ecuaciones de la matriz triangula inferior 

        # ---> LUx = b entonces L(Ux) = b, si y = Ux, entonces Ly = b porque U * x es un vector 
        
        # L * y = e_i 
        y = res_tri(L, e_i, inferior=True)

        # Y ahora U * x = y
        x = res_tri(U, y, inferior=False)

        A_inversa[:, i] = x # llenando columna por columna, vector por vector 

    return A_inversa

         
    
def calculaLDV(A):
    # Si A = LU y quiero A = LDV, entonces U = DV, siendo D la diagonal y V la triangular superior, con unos como diagonal 

    L, U, _ = calculaLU(A)

    if L is None:
        return None, None, None

    n = np.shape(A)[0]
    # Base de contrucción 
    D = np.zeros((n, n))
    V = np.zeros((n, n)) 

    for i in range(n):

        # no puede haber elementos de la diagonal que sean 0
        if U[i,i] == 0:
            return None, None, None 

        D[i,i] = U[i,i]

        V[i,i:] = U[i,i:] / U[i,i] ## como U = DV, entonces, Uij = Dii * V ij --> Vij = Uij/Uii --> siendo tmb Uii la diagonal 

    return L, D, V



# Simétrica Definida positiva <=> x^t * A * x > 0 para todo x != 0  And   A^t = A 
# Si A = LDV y A = A^t ----> V = L^t ---> A = LDL^t 
#  --------> x^t * LDL^t * x > 0 ----> (x^t * L) * D * (x * L^t) > 0  ------> y^t * D * y > 0, pero D es solo una diagonal con cosas
#  ------> si se hace esta multiplicación, quedará definida como una sumatoria desde i=1 hasta n de Dii yi^2
# ---------> pero yo quería ver que todo esto es > 0, and yi^2 es mayor que 0 siempre si y !=0, entonces me basta con chequear el signo de D 

def esSDP(A, atol=1e-8):
    A_arr = np.array(A) * 1.0

    n = np.shape(A_arr)[0]
    m = np.shape(A_arr)[1]
    if m != n:
        return False

    # nos plantamos desde la diagonal a la derecha, así chequeamos la simetría con su contraparte 
    for i in range(n):
        for j in range(i + 1, n):   
            diferencia = A_arr[i, j] - A_arr[j, i]
            if diferencia > atol or diferencia < -atol:
                return False # si no es simétrica, chau

                
    L, D, _ = calculaLDV(A_arr)

    #defensa pivote 0
    if L is None:
        return False

    # Chequeando la positividad de D 
    for i in range(n):
        if D[i, i] <= atol:
            return False
            
    return True


def calculaCholesky(A, atol=1e-10):

    if not esSDP(A, atol):
        return None
    
    n = np.shape(A)[0]
    R = np.zeros((n, n))

    # algoritmo Cholesky
    # para cada columna
    for j in range(n):
        # a la derecha de la diagonal, incluyendola
        for i in range(j, n):
            suma = 0.0
            for k in range(j):
                suma += R[i, k] * R[j, k]

            # Literalmente el algoritmo que dijo Lore 
            if i == j:
                R[j, j] = (A[j, j] - suma) ** 0.5
            else:
                R[i, j] = (A[i, j] - suma) / R[j, j]
    return R


# Funciones del labo 5 --------------------------------


def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones).
    Si la matriz A no es de n x n, debe retornar None
    """

    dimensiones = np.shape(A)
    if dimensiones[0] != dimensiones[1]:
        return None
    
    n = dimensiones[0]
    
    Q_cols = [] 
    R_matrix = np.zeros((n, n))
    
    # for sobre A
    for j in range(n):
        # para toda colum j
        v_j = A[:, j].copy() 

        # Proyección
        # para todos los q anteriores
        # para iteraciones luego de q_1 
        for k in range(len(Q_cols)):
            # esa q ya calculada 
            q_k = Q_cols[k]

            # producto interno, como asumo que q ya debería estar normalizada, esta r es directamente la proyección de v sobre q
            r_kj = np.sum(q_k * v_j)
            
            # y la guardamos en la matriz R triangular superior 
            R_matrix[k, j] = r_kj

            # ahora la nueva q solo le falta estar normalizada
            v_j = v_j - (r_kj * q_k)


        # Norma 2, es decir, la escala, esta se guarda en la diagonal de R la cual no puede ser 0  
        r_jj = np.sqrt(np.sum(v_j * v_j))
            
        # tolerancia de 0
        if r_jj > tol:
            i = len(Q_cols) 
            R_matrix[i, j] = r_jj
            
            # Ahora si se normaliza y se guarda 
            q_j = v_j / r_jj
            Q_cols.append(q_j) # todas las q de Q_cols están normalizadas

    r = len(Q_cols)
    
    if r > 0:
        Q = transponer(Q_cols, n)
    else:
        Q = np.zeros((n, 0)) 

    # No mas LD
    R = R_matrix[:r, :] 
    
    if retorna_nops:
        return Q, R, 0
    
    return Q, R


def QR_con_HH(A,tol=1e-12,extras=False):
    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m>=n, debe retornar None
    extras : bool, opcional
        Si es True, devuelve informacion extra sobre el proceso de factorizacion.
        Por defecto es False. Esto lo hacemos para poder graficar el proceso.
    Devuelve la factorizacion QR de A usando reflectores de Householder.
    Devuelve: 
        Q, R, extra_info (si extras es True)
        Q, R (si extras es False)
    extra_info es un diccionario con la clave:
        'R_matrices': lista de las matrices R en cada paso
        'Q_matrices': lista de las matrices Q en cada paso
    """

    dimensiones = np.shape(A)
    if len(dimensiones) != 2:
        return None
        
    m = dimensiones[0]
    n = dimensiones[1]
    
    if m < n:
        return None
        
    R = np.copy(A)
    Q = np.eye(m)
    
    if extras:
        R_matrices = []
        Q_matrices = []

    for k in range(n):
        
        x = R[k:m, k]
        
        # Norma euclidiana
        norm_x = np.sqrt(np.sum(x * x))
        
        # Determinación de signo opuesto (para la contrucción de u) para olvidarme del problema 
        # de los números flotantes 
        signo = 1.0 if x[0] >= 0 else -1.0
        alpha = -signo * norm_x
        
        # vector director u
        # fuerzo el vector de destino y para que y = (alpha, 0, 0 ... 0)
        # donde |alpha| = ||x|| ---> por esto no me importa el signo, así que lo fuerzo a conveniencia
        
        u = np.copy(x)
        # cálculo implícito de u = x - y, siendo y el vector destino 
        u[0] = u[0] - alpha # ----> la operación u[0] - alpha será siempre adición, por tanto, no molestarán los flantes
        
        # norma de u 
        norm_u = np.sqrt(np.sum(u * u))
        
        if norm_u > tol:
            u = u / norm_u

            # Contrucción de R a través del registro 
            # acumulativo de reflexiones desde la matriz A
            # multiplicando por izquierda
            # Hn ... H1 A = R
            # R = R - 2 * u * (u^T * R)
            # OBS --> H ortogonal, entonces H^-1 = H^t = H 
            for j in range(k, n): 
                dot_prod = np.sum(u * R[k:m, j])
                R[k:m, j] = R[k:m, j] - 2.0 * dot_prod * u # R = H* ...H A

            # Contrucción de Q a través del registro 
            # acumulativo de todas las refleciones inversas 
            # desde la matriz identidad multiplicando por derecha 
            # Q Hn ... H1 = R
            # Q = Q - 2 * (Q * u) * u^T

            for i in range(m):
                dot_prod = np.sum(Q[i, k:m] * u)
                Q[i, k:m] = Q[i, k:m] - 2.0 * dot_prod * u
                
        # Extras
        if extras:
            R_matrices.append(np.copy(R))
            Q_matrices.append(np.copy(Q))

    if extras:
        extra_info = {'R_matrices': R_matrices, 'Q_matrices': Q_matrices}
        return Q, R, extra_info
        
    return Q, R

def calculaQR(A,metodo='RH',tol=1e-12):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R    
    metodo = ['RH','GS'] usa reflectores de Householder (RH) o Gram Schmidt (GS) para realizar la factorizacion
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones)
    Si el metodo no esta entre las opciones, retorna None
    """
    if metodo == 'RH':
        return QR_con_HH(A, tol=tol)
    elif metodo == 'GS':
        return QR_con_GS(A, tol=tol)
    else:
        return None


# Funciones del labo 6 --------------------------------

rng = np.random.default_rng()


def metpot2k(A, tol=1e-15, K=1000):
    A = np.array(A)

    n = np.shape(A)[0]
    m = np.shape(A)[1]

    if n != m: 
        return None


    vector = np.random.random(size=n)
    
    vectorMonho = funcTL(A, vector, 2)

    error = arroba(transpuesta(vectorMonho), vector)[0]

    k = 0 

    while abs(error - 1) > tol and k < K: 
        vector = vectorMonho 

        vectorMonho = funcTL(A, vector, 2)

        error = arroba(transpuesta(vectorMonho), vector)[0]

        k = k + 1 

    lamb = arroba(arroba(transpuesta(vectorMonho), A), vectorMonho)[0]

    e = error - 1 

    return vectorMonho, lamb, k


def diagRH(A, tol=1e-15, K=1000):
    n = np.shape(A)[0]
    
    v1, l1, _ = metpot2k(A, tol, K)
    
    e1 = np.zeros(n)
    e1[0] = 1.0
    
    u = e1 - v1
    
    norma_u_cuad = norma(u, 2)**2
    
    Hv1 = np.zeros((n, n))
    for i in range(n):
        Hv1[i, i] = 1.0  
        
    if norma_u_cuad >= 1e-15:
        for i in range(n):
            for j in range(n):
                Hv1[i, j] -= 2 * u[i] * u[j] / norma_u_cuad

    # Caso base
    if n == 2:
        S = Hv1
        # B = Hv1 @ A
        B = arroba(Hv1, A)
        # D = B @ Hv1^T
        D = arroba(B, transpuesta(Hv1))
        return S, D
        
    # Paso Recursivo (n > 2)
    else:
        # H * A * H^T
        B_temp = arroba(Hv1, A)
        B = arroba(B_temp, transpuesta(Hv1))
        
        A_tilde = B[1:, 1:]
        
        # Llamada recursiva
        S_tilde, D_tilde = diagRH(A_tilde, tol, K)
        
        D = np.zeros((n, n))
        D[0, 0] = l1
        for i in range(n - 1):
            for j in range(n - 1):
                D[i+1, j+1] = D_tilde[i, j]
        
        S_aux = np.zeros((n, n))
        S_aux[0, 0] = 1.0
        for i in range(n - 1):
            for j in range(n - 1):
                S_aux[i+1, j+1] = S_tilde[i, j]
        
        S = arroba(Hv1, S_aux)
        
        return S, D


# Funciones del labo 7 --------------------------------



def transiciones_al_azar_continuas(n):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    Retorna matriz T de nxn normalizada por columnas, y con entradas al azar en el
    intervalo [0,1]
    """

    T = np.random.random((n, n))
    for j in range(n):
        suma_columna = 0.0
        
        for i in range(n):
            suma_columna += T[i, j]
        
        if suma_columna > 0:
            for i in range(n):
                T[i, j] = T[i, j] / suma_columna
                
    return T
    
def transiciones_al_azar_uniformes(n,thres):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    thres probabilidad de que una entrada sea distinta de cero.
    Retorna matriz T de n x n normalizada por columnas. 
    El elemento i,j es distinto de cero si el número generado al azar para i,j es menor o igual a thres. 
    Todos los elementos de la columna $j$ son iguales 
    (a 1 sobre el número de elementos distintos de cero en la columna).
    """
    T = np.random.random((n, n))

    for j in range(n):

        suma_columna = 0.0
        
        for i in range(n):
            if T[i,j] < thres:
                T[i,j] = 0.0
            else:
                T[i,j] = 1.0

            suma_columna += T[i, j]
        
        if suma_columna > 0:
            for i in range(n):
                T[i, j] = T[i, j] / suma_columna
        else:
            for i in range(n):
                T[i,j] = 1.0 / n
        
    return T


def nucleo(A,tol=1e-15):
    """
    A una matriz de m x n
    tol la tolerancia para asumir que un vector esta en el nucleo.
    Calcula el nucleo de la matriz A diagonalizando la matriz traspuesta(A) * A (* la multiplicacion matricial), usando el medodo diagRH. El nucleo corresponde a los autovectores de autovalor con modulo <= tol.
    Retorna los autovectores en cuestion, como una matriz de n x k, con k el numero de autovectores en el nucleo.
    """
    n = np.shape(A)[1]
    
    At = transpuesta(A)
    AtA = arroba(At, A)
    
    S, D = diagRH(AtA)
    
    autovectores_validos = []
    for i in range(n):
        if abs(D[i, i]) <= tol:
            autovectores_validos.append(S[:, i])
            
    k = len(autovectores_validos)

    if k == 0:
        return np.array([])

    if k == 0:
        return np.zeros((n, 0))
        
    res = np.zeros((n, k))
    for j in range(k):
        for i in range(n):
            res[i, j] = autovectores_validos[j][i]
            
    return res
    
        
        

def crea_rala(listado,m_filas,n_columnas,tol=1e-15):
    """
    Recibe una lista listado, con tres elementos: lista con indices i, lista con indices j, y lista con valores A_ij de la matriz A. Tambien las dimensiones de la matriz a traves de m_filas y n_columnas. Los elementos menores a tol se descartan.
    Idealmente, el listado debe incluir unicamente posiciones correspondientes a valores distintos de cero. Retorna una lista con:
    - Diccionario {(i,j):A_ij} que representa los elementos no nulos de la matriz A. Los elementos con modulo menor a tol deben descartarse por default. 
    - Tupla (m_filas,n_columnas) que permita conocer las dimensiones de la matriz.
    """
    if len(listado) < 3:
        return [{}, (m_filas, n_columnas)]
        
    lista_i = listado[0]
    lista_j = listado[1]
    lista_v = listado[2]
    
    diccionario_rala = {}
    
    for k in range(len(lista_v)):
        valor = lista_v[k]
        
        if abs(valor) >= tol:
            i = lista_i[k]
            j = lista_j[k]
            diccionario_rala[(i, j)] = valor
            
    return [diccionario_rala, (m_filas, n_columnas)]


def multiplica_rala_vector(A,v):
    """
    Recibe una matriz rala creada con crea_rala y un vector v. 
    Retorna un vector w resultado de multiplicar A con v
    """
    diccionario_rala = A[0]
    dimensiones = A[1]
    m_filas = dimensiones[0]
    
    # El vector resultado tendrá el tamaño de las filas de A
    w = np.zeros(m_filas)
    
    # Multiplicación optimizada: solo iteramos sobre los elementos distintos de cero
    for (i, j), valor in diccionario_rala.items():
        w[i] += valor * v[j]
        
    return w



# Funciones del labo 8 --------------------------------



def svd_reducida(A, k="max", tol=1e-15):
    """
    A la matriz de interes (de m x n)
    k el numero de valores singulares a retener.
    tol la tolerancia para considerar un valor singular igual a cero
    """
    m, n = np.shape(A)
    transpuesto = False
    
    # si m < n, inversa 
    if m < n:
        A = transpuesta(A)
        m, n = np.shape(A)
        transpuesto = True
        
    # Garantía de m >= n
    
    # Diagonalizar A^T A
    At_A = arroba(transpuesta(A), A)
    V, D = diagRH(At_A, tol, K=1000)
    
    # tolerancia a prtir de autovalores
    sigmas = []
    for i in range(n):
        autovalor = D[i, i]
        if autovalor > tol:
            sigma = autovalor ** 0.5
            sigmas.append(sigma)
            
    # Rango a partir de los sigmas 
    r = len(sigmas)
    if k != "max":
        r = min(r, k)
        
    hatSig = np.array(sigmas[:r])
    
    # Matriz V
    hatV = np.zeros((n, r))
    for i in range(n):
        for j in range(r):
            hatV[i, j] = V[i, j]
            
    # Matriz U a partir ed V normalizando la matriz B 
    hatU = np.zeros((m, r))
    B = arroba(A, hatV)
    
    for j in range(r):
        for i in range(m):
            hatU[i, j] = B[i, j] / hatSig[j]

    # final general 
    if transpuesto:
        return hatV, hatSig, hatU
    else:
        return hatU, hatSig, hatV