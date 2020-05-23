# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 21:39:03 2017

@author: Jorge Hermosillo

Modelo lineales:
    - Modelo basico
    - Perceptron dual
    - Perceptron con kernel
        - kernel lineal
        - kernel gaussiano
        - kernel polinomial
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import train_test_split

def datos_lin_separables(clusters=2,samples=100):
    """
    Genera datos linealmente separables
    Se generan hasta 10 clusters distintos c/u con distribución normal N(mu,s)
    Entrada: numero de clusters y numero de muestras por cluster (clase)
    Salida: Lista con matrices de vectores de datos y código de clase por cluster.
    """
    if clusters > 10:
        n = 10
        print('solo se pueden producir hasta 10 clusters')
    elif clusters < 2:
        n = 2
        print('el numero minimo de clusetrs es 2')
    else:
        n = clusters
    
    """    
    arreglo que contiene los valores promedio de cada distribucion de datos
    """
    mus= []   
    
    if n == 2:
        mus.append([-3,-3])
        mus.append([3,3])
    elif n == 3:
        mus.append([-3,-3])
        mus.append([0,3])
        mus.append([5,-3])
    elif n == 4:
        mus.append([-5,-3])
        mus.append([-3,3])
        mus.append([4,-3])
        mus.append([5,4])
    elif n == 5:
        mus.append([-5,-3])
        mus.append([-3,4])
        mus.append([4,-3])
        mus.append([5,4])
        mus.append([1,1])
    elif n == 6:
        mus.append([-5,-3])
        mus.append([-3,4])
        mus.append([4,-3])
        mus.append([5,4])
        mus.append([1,1])
        mus.append([-9,3])
    elif n == 7:
        mus.append([-5,-3])
        mus.append([-3,4])
        mus.append([4,-3])
        mus.append([5,4])
        mus.append([1,1])
        mus.append([-9,3])
        mus.append([8,-3])
    elif n == 8:
        mus.append([-5,-3])
        mus.append([-3,4])
        mus.append([4,-3])
        mus.append([5,4])
        mus.append([1,1])
        mus.append([-9,3])
        mus.append([8,-3])
        mus.append([12,6])
    elif n == 9:
        mus.append([-5,-3])
        mus.append([-3,4])
        mus.append([4,-3])
        mus.append([5,4])
        mus.append([1,1])
        mus.append([-9,3])
        mus.append([8,-3])
        mus.append([12,6])
        mus.append([-2,8])
    elif n == 10:
        mus.append([-5,-3])
        mus.append([-3,4])
        mus.append([4,-3])
        mus.append([5,4])
        mus.append([1,1])
        mus.append([-9,3])
        mus.append([8,-3])
        mus.append([12,6])
        mus.append([-2,8])
        mus.append([11,0])
        
    mus = np.asarray(mus)
    
    """    
    valor de correlación (rho) entre los datos de una distribución
    """
    rho = np.random.uniform(low=0.2, high=0.8, size=(n,))  
    
    if n==2:
        p=np.random.rand()
        if p < 0.5:
            rho[0]=-rho[0]
            rho[1]=rho[1]
    if n > 3:
        for i in range(n):
            rho[i] *= (-1)**i
    
    """    
    Desviaciones estandar (s) de cada cluster de datos
     - Se genera un vector de dimensión 2xn con valores de s= 1 + un valor aleatorio
     - Luego se transforma el arreglo en una matriz bidimensional con n filas y 2 columnas
       donde n es el número de clusters
    """
    s = np.random.rand(2*n)+1
    s = s.reshape((n,2))
    
    """    
    Matriz de covarianzas
     - Se construye la matriz:
      [[s_0^2   s_0 x s_1 x rho]
       [s_0 x s_1 x rho   s_1^2]]
    """
    cov_list = []
    for i,v in enumerate(s):
        cov = np.zeros((2,2))
        cov[0,0] = v[0]**2
        cov[1,1] = v[1]**2
        cov[0,1] = v[0]*v[1]*rho[i]
        cov[1,0] = cov[0,1]
        cov_list.append(cov)
    
    """    
    Arreglo de salida
    """
    L = []
    
    clases = np.arange(n,dtype='int')
    if n==2:
        clases[0]=-1
        
    """    
    Se genera un arreglo con el valor de clase por el número de muestras
    """
    y = np.repeat(clases[0],samples).reshape(samples,1).astype(int)
    
    for i in range(1,n):
        yp = np.repeat(clases[i],samples).reshape(samples,1).astype(int)
        y = np.hstack((y,yp))
    
    """    
    Se genera un arreglo de clusters, que son distribuciones normales incluyendo el valor de la clase 
    """
    for i in range(n):
        X = np.random.multivariate_normal(mus[i], cov_list[i],samples)
        X = np.hstack((X,y[:,i].reshape(samples,1).astype(int)))
        L.append(X)
        
    return L

def split_train_test(X,y,Tp):
    """
    Obten datos de entrenamiento y prueba
    Entrada: datos centrados, clases (+/-1) y proporcion de entrenamiento (Tp)
    Salida: valores de entrenamiento y prueba
    """
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    for i in range(len(X)):
        X1 = X[i]
        y1 = y[i]
        X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, train_size=Tp/100.,shuffle=True)
        X_train.append(X1_train)
        X_test.append(X1_test)
        y_train.append(y1_train)
        y_test.append(y1_test)    
    
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    X_train = X_train.ravel().reshape(-1,2)
    y_train = y_train.ravel()
    X_test = X_test.ravel().reshape(-1,2)
    y_test = y_test.ravel()
    return X_train, X_test, y_train, y_test
  
def datos_solapados(n=2,samples=100):
    """
    Genera datos solapados (no linealmente separables)
    Entrada: numero de muestras
    Salida: vectores de datos clase 1 y 2.
    """    
    #Gaussiana 1
    mu1=np.array([3.0,5.0])
    s1=np.array([1.0,2.5])
    #matriz de covarianzas centrada en mu
    corr=0.42
    co_v= s1[0]*s1[1]*corr
    cov1 = [[s1[0]**2,co_v],[co_v,s1[1]**2]]
    
    mu2=np.array([4.0,4.0])
    s2=np.array([0.8,1.7])
    #matriz de covarianzas centrada en mu
    corr=0.3
    co_v= s2[0]*s2[1]*corr
    cov2 = [[s2[0]**2,co_v ],[co_v, s2[1]**2]]
    
    cov_list = [cov1,cov2]
    mus = [mu1,mu2]

    L = []
    
    clases = np.arange(2,dtype='int')
    clases[0]=-1
    y = np.repeat(clases[0],samples).reshape(samples,1).astype(int)
    for i in range(1,2):
        yp = np.repeat(clases[i],samples).reshape(samples,1).astype(int)
        y = np.hstack((y,yp))
    
    for i in range(2):
        X = np.random.multivariate_normal(mus[i], cov_list[i],samples)
        X = np.hstack((X,y[:,i].reshape(samples,1).astype(int)))
        L.append(X)
        
    return np.array(L)

def centra_datos(X):
    """
    Elimina promedios de los datos por feature X[:,j]
    Entrada: matriz de datos X
    Salida: matriz de datos centrada Xc
    """
    if X.ndim != 1:
        Xm=[X[:,j].mean() for j in range(X.shape[1])]
        Xm=np.asarray(Xm)
        Xc=X-Xm
    else:
        Xc=X-X.mean()
    return Xc


def toy_data(tipo='ls',n=2,s=100,prop=80):
    if tipo=='ls':
        L=datos_lin_separables(samples=s)
    else:
        L=datos_solapados(samples=s)
    
    #Listas de datos y clases
    X=[]
    y=[]
    for i in range(len(L)):
        X.append(L[i][:,:-1])
        y.append(L[i][:,-1].astype(int))

    """Separa los datos en entrenamiento y prueba"""
    #80% de datos para entrenar
    x_train,x_test,y_train,y_test=split_train_test(X,y,prop) 

    """Centra los datos"""
    x_train=centra_datos(x_train)
    x_test=centra_datos(x_test)
    
    clases = np.arange(n,dtype='int')
    clases[0]=-1

    return x_train,x_test,y_train,y_test,clases

def grafica(x_train,y_train,x_test,y_test):
    """Areas de Graficacion y visualizacion de los datos"""
    fig,ax = plt.subplots(figsize=(6,5))
    
    clases = np.unique(y_train).astype(int)

    """Datos negativos de entrenamiento"""
    ax.scatter(x_train[:,0][y_train==clases[0]], x_train[:,1][y_train==clases[0]],
               facecolor='royalblue', 
               marker='$\\ominus$', 
               edgecolor='royalblue',
               s=80,
               label='neg train')
    """Datos positivos de entrenamiento"""
    ax.scatter(x_train[:,0][y_train==clases[1]], x_train[:,1][y_train==clases[1]],
               facecolor='orangered', 
               marker='$\\bigoplus$', 
               edgecolor='orangered',
               s=80,
               label='pos train')

    """Datos de PRUEBA"""
    xmin = np.amin(x_train[:,0])+0.05*np.amin(x_train[:,0]) 
    xmax = np.amax(x_train[:,0])+0.05*np.amax(x_train[:,0]) 
    ymin = np.amin(x_train[:,1])+0.05*np.amin(x_train[:,1]) 
    ymax = np.amax(x_train[:,1])+0.05*np.amax(x_train[:,1]) 
    ax.axis([xmin,xmax,ymin,ymax])

    ax.scatter(x_test[:,0][y_test==clases[0]], x_test[:,1][y_test==clases[0]],
               facecolor='k', 
               marker='o', 
               edgecolor='k',
               s=45,
               label='neg test')
    ax.scatter(x_test[:,0][y_test==clases[1]], x_test[:,1][y_test==clases[1]],
               facecolor='r', 
               marker='o', 
               edgecolor='k',
               s=45,
               label='pos test')
    """Parametros de la clave"""
    legend = plt.legend(loc='upper center',
                    bbox_to_anchor=(0.5, 1.075),
                    ncol=4,
                    fancybox=True,
                    shadow=False)

    return fig,ax

def plot_FD(fig,ax,clf,show_SV=False):
    """
    Calcula el contorno de una curva 
    que representa la frontera de decision de un perceptron con kernel no-lineal
    Entrada: apuntador al modelo y zona de graficación
    Salida: curva de contorno (grafico)
    """
    #     vectores de soporte
    if show_SV:
        ax.scatter(clf.sv[:,0], clf.sv[:,1], 
                   s=50, 
                   facecolor='none',
                   edgecolor="g",
                   linewidth=3, 
                   zorder=10,
                   label='SV_KERNEL')
    if clf.__class__.__name__=='SVC':
        c='green'
        clave='SVM'
    else:
        c='purple'
        clave=clf.clave
    xmin,xmax=ax.get_xlim()
    ymin,ymax=ax.get_ylim()
    
    X1, X2 = np.meshgrid(np.linspace(xmin,xmax,100), np.linspace(ymin,ymax,100))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = clf.decision_function(X)
    Z = Z.reshape(X1.shape)
    CS=ax.contour(X1, X2, Z, 0,
                  colors=c, 
                  linestyles='dashed',
                  linewidths=3,
                  origin='lower')
    ax.contourf(X1, X2, Z,alpha=0.2,levels=0,cmap=cm.jet)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    CS.collections[0].set_label(clave)
    """Parametros de la clave"""
    legend = plt.legend(loc='upper center',
                        bbox_to_anchor=(0.5, 1.15),
                        ncol=4,
                        fancybox=True,
                        shadow=False)

    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_alpha(1)
    fig.tight_layout()

    return

def plot_FD_SVM(fig,ax,clf,SVP=False):
    """
    Calcula el contorno de una curva 
    que representa la frontera de decision de SVM
    Entrada: apuntador al modelo y zona de graficación
    Salida: curva de contorno (grafico)
    """
    #     vectores de soporte
    p=clf.get_params()
    print()

    if p['kernel']=='linear':
        if SVP:
            #frontera de decision
            w = clf.coef_[0]
            a = -w[0] / w[1]
            xx = np.linspace(-5, 5)
            yy = a * xx - (clf.intercept_[0]) / w[1]

            """
            plot the parallels to the separating hyperplane that pass through the
            support vectors (margin away from hyperplane in direction
            perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
            2-d.
            """
            margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
            yy_down = yy - np.sqrt(1 + a ** 2) * margin
            yy_up = yy + np.sqrt(1 + a ** 2) * margin

            # plot the line, the points, and the nearest vectors to the plane
            ax.plot(xx, yy, color='maroon',ls='-',label='SVM')
            ax.plot(xx, yy_down, 'b--',label='m-')
            ax.plot(xx, yy_up, 'r--',label='m+')
            
            ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                       s=80,
                       facecolor='none',
                       zorder=10,
                       edgecolor='g',
                       lw=3,
                       label='SV_SVM')

    if clf.__class__.__name__=='SVC':
        c='green'
        clave='SVM'
    else:
        c='purple'
        clave=clf.clave
    xmin,xmax=ax.get_xlim()
    ymin,ymax=ax.get_ylim()
    
    X1, X2 = np.meshgrid(np.linspace(xmin,xmax,100), np.linspace(ymin,ymax,100))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = clf.decision_function(X)
    Z = Z.reshape(X1.shape)
    CS=ax.contour(X1, X2, Z, 0,
                  colors=c, 
                  linestyles='dashed',
                  linewidths=3,
                  origin='lower')
    ax.contourf(X1, X2, Z,alpha=0.2,levels=0,cmap=cm.jet)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    CS.collections[0].set_label(clave)
    """Parametros de la clave"""
    legend = plt.legend(loc='upper center',
                        bbox_to_anchor=(0.5, 1.15),
                        ncol=4,
                        fancybox=True,
                        shadow=False)

    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_alpha(1)
    fig.tight_layout()

    return

class modelo_basico(object):
    """
    Modelo basico de clasificacion binaria
    """
    def __init__(self,clases):
        self.w = np.zeros(2)  #vector de pesos inicializado en ceros
        self.clases = clases
    
    def fit(self, X_train,y_train):
        """
        Funcion de aprendizaje
        Entrada: datos de entrenamiento X,y
        Salida: valores de la frontera de decision
        Variables internas: 
            - w: pesos (fronteras de decision)            
        """ 
        #centroides de los datos
        m_neg=np.array([X_train[:,0][y_train==self.clases[0]].mean(),
                        X_train[:,1][y_train==self.clases[0]].mean()])
        m_pos=np.array([X_train[:,0][y_train==self.clases[1]].mean(),
                        X_train[:,1][y_train==self.clases[1]].mean()])
        
        #visualizacion de los centroides
        plt.scatter([m_pos[0]],[m_pos[1]],
                    facecolor='r',
                    edgecolor='k',
                    marker='x',
                    s=120,
                    label='centroide pos')
        plt.scatter([m_neg[0]],[m_neg[1]],
                    facecolor='b',
                    edgecolor='k',
                    marker='x',
                    s=120,
                    label='centroide neg')
        
        plt.plot([m_neg[0],m_pos[0]],[m_neg[1],m_pos[1]],
                 color='olive',
                 ls='--',
                 lw=1)

        #vector de pesos
        self.w = m_pos - m_neg
        
        #vector de la frontera de decision (recta)
        wT = np.array([-self.w[1],self.w[0]])
    
        #punto medio entre los centroides
        mX=m_neg[0]/2+m_pos[0]/2
        mY=m_neg[1]/2+m_pos[1]/2
        centro=np.array([mX,mY])
        
        #valores de delimitacion de la imagen
        xmin = np.amin(X_train[:,0])+0.1*np.amin(X_train[:,0]) 
        xmax = np.amax(X_train[:,0])+0.1*np.amax(X_train[:,0])
        
        #visualizacion de la frontera
        self.x, self.y = plot_linea(centro, wT, xmin, xmax)

        return self.x, self.y
    
    def predict(self,X):
        """
        Funcion de prediccion
        Entrada: arreglo numpy con datos de entrenamiento/prueba X
        Salida:  arreglo (unidimensional) y 
                 con valores de prediccion de clase (+1 , -1)          
        """ 
        return np.sign(np.dot(X,self.w))

    def metricas(self, y_test, y_predicted):
        """
        Funcion de evaluacion del rendimiento
        Entrada: arreglos numpy con datos de prediccion de clase y de prueba
        Salida: dos valores reales: exactitud y precision          
        """ 
        #r contiene los indices donde la prediccion 
        #y los valores reales de clase coinciden
        r = np.where(y_predicted==y_test)
        
        #y_p contiene los indices donde la prediccion es pos
        y_p = np.where(y_predicted>0)
        
        #r_p contiene los datos (+1 o -1) de y_test indizados por y_p 
        rp = y_test[y_p]
        
        #rp_t conserva solo los valores de rp que son pos 
        rp_t = rp[rp>0]
        
        #metricas de desepenio exactitud y precision
        exact= len(r[0])/len(y_test)*100
        prec = len(rp_t)/len(rp)*100
        
        return exact, prec
    
class Perceptron(object):
    """
    Clasificador Perceptron
    """

    def __init__(self, epocas=1, eta=1.):
        self.epocas = epocas    #epocas: 1 por defecto
        self.eta=eta  #tasa de aprendizaje: 1 por defecto

    def fit(self, X, y,show_error=False):
        """
        Funcion de aprendizaje
        Entrada: datos de entrenamiento X,y en coordenadas homogeneas
        Calcula: W, los pesos del modelo, y el Error de aprendizaje
        """         
        n_samples, d_features = X.shape                 #matriz de datos
        self.W = np.zeros(d_features, dtype=np.float64) #vector de pesos en cero
        self.W_history=[]
        errors = []

        for e in range(self.epocas):
            total_error = 0
            for i in range(n_samples):
                if self.predict(X[i])*y[i] <= 0:
                    self.W_history.append(self.W)
                    self.W += y[i] * self.eta * X[i]
                    total_error += self.predict(X[i])*y[i]
            errors.append(total_error)
        if show_error:
            xticks=np.arange(self.epocas)
            plt.xticks(xticks,[str(x) for x in xticks])
            plt.plot(errors,label='Perceptron')
            plt.xlabel('Epoca')
            plt.ylabel('Error total')
            plt.legend()

        
    def predict(self, X):
        """
        Funcion que determina el signo de la proyeccion
        Entrada: arreglo numpy de datos X
        Salida:  arreglo numpy con valores +1 o -1 
        """         
        return np.sign(X@self.W).astype(int)
    
    def metricas(self, y_test, y_predicted):
        """
        Funcion de evaluacion del rendimiento
        Entrada: arreglos numpy con datos de prueba y prediccion
        Salida: exactitud, precision y recall          
        """ 
        #r contiene los indices donde la prediccion y los valores reales coinciden
        r = np.where(y_predicted==y_test)
        acc = y_test[r]
        
        y_p = np.where(y_predicted>0)
        pos=np.where(y_test>0)
        
        FP = set(list(y_p[0])).symmetric_difference(set(list(pos[0])))
        TP = set(list(pos[0]))-FP        
        TP_FP = TP.union(FP) 
                
        #exactitud, precision y sensibilidad
        accuracy= len(acc)/len(y_test)*100
        precision = len(list(TP))/len(list(TP_FP))*100
        recall = len(list(TP))/len(pos[0])*100
        
        return accuracy, precision, recall
    
    def f(self,x,c=0):
        """
        dados x y w, regresa y tal que [x,y] esta sobre la linea
        w.x + b = c
        w es el vector ortogonal a la recta
        """
        v=(-self.W[1:][0] * x - self.W[0] + c) / self.W[1:][1]
        return v

    
    def plot_FD(self,ax):
        """
        Calcula los puntos de la linea recta 
        que representa la frontera de decision de un perceptron lineal
        Entrada: datos de las dos clases X1 y X2
        Salida: dos puntos extremos de la frontera de decision
        """
        xmin,xmax=ax.get_xlim()
        ymin,ymax=ax.get_ylim()

        a0 = xmin
        a1 = self.f(a0)

        b0 = xmax 
        b1 = self.f(b0)
        
        x=[a0,b0]
        y=[a1,b1]
        ymin,ymax=plt.ylim()
        ax.axis([x[0],x[1],ymin,ymax])
        ax.plot(x, y, "k", label='Perceptron')
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)
        #leyenda
        legend = plt.legend(loc='upper center',
                    bbox_to_anchor=(0.5, 1.15),
                    ncol=4,
                    fancybox=True,
                    shadow=False)

        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_alpha(1)
        return

class PerceptronKernel():
    """
    Perceptron con Kernel básico
    """
    def __init__(self, kernel='lk',**params):
        self.epocas=1
        if kernel == 'lk':
            self.__kernel=self.__lk
            self.clave = 'K-Lineal'          #para la leyenda de la imagen
        elif kernel == 'gk':
            self.__kernel=self.__gk
            self.clave = 'K-Gauss'
        else:
            self.__kernel=self.__pk
            self.clave = 'K-Poli'
        if params:
            for p in params.keys():
                if p=='epocas':
                    self.epocas = params[p]
                    continue
                if p=='gamma':
                    self.gamma=params[p]

    def __lk(self,Xi,Xj):
        return Xi@Xj
    
    def __pk(self,Xi,Xj):
        return (1 + Xi@Xj)**self.gamma

    def __gk(self,Xi, Xj):
        return np.exp(-self.gamma*np.linalg.norm(Xi-Xj)**2)

    @property    
    def kernel(self):
        return self.__kernel
    
    @kernel.setter
    def kernel(self,*args):
        try:
            l=args[0]
            ok=True
            if isinstance(l,str):
                kernel=l
            else:
                kernel=l[0]
                try:
                    parametro=l[1]
                except IndexError:
                    print("ERROR: Debes pasar también un parámetro-> gamma > 0")
                    ok=False
        except IndexError:
            print("ERROR: Debes pasar al menos un argumento con la etiqueta del kernel: {lk,gk,pk}")
            ok=False
        if ok:
            if kernel == 'lk':
                self.__kernel=self.__lk
                self.clave = 'K-Lineal'          #para la leyenda de la imagen
            elif kernel == 'gk':
                self.gamma = parametro       #gamma del kernel gaussiano
                self.__kernel=self.__gk
                self.clave = 'K-Gauss'
            else:
                self.gamma = parametro       #gamma del kernel gaussiano
                self.__kernel=self.__pk
                self.clave = 'K-Poli' 
        return

    def fit(self, X, y):
        """
        Funcion de aprendizaje
        In: X, matriz de instancias de aprendizaje
            y, matriz de clases
        Out: alpha vector de pesos de instancias de aprendizaje
        """         
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples, dtype=np.float64)
        
        clases = np.unique(y).astype(int)
        if clases[0]==0:
            y_temp=y[:]
            y_temp[y==0]=-1
            self.__y=y_temp[:]
            y_temp=[]
        else:
            self.__y=y[:]

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        # Entrenamiento
        for epoca in range(self.epocas):
            for i in range(n_samples):
                if self.__y[i]*np.sign(np.sum(K[i,:] * self.alpha * self.__y))<=0:
                    self.alpha[i] += 1.0

        # Support vectors
        sv = self.alpha > 1e-5
        ind = np.arange(len(self.alpha))[sv]
        self.alpha = self.alpha[sv]
        self.sv = X[sv]
        self.sv_y = self.__y[sv]
        print("{0} vectores de soporte de {1} puntos".format(len(self.alpha),n_samples))

    def decision_function(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.alpha, self.sv_y, self.sv):
                s += a * sv_y * self.__kernel(X[i], sv)
            y_predict[i] = s
        return y_predict

    def predict(self, X):
        return np.sign(self.decision_function(X))
    
    
    def metricas(self, y_test, y_predicted):
        """
        Funcion de evaluacion del rendimiento
        Entrada: arreglos numpy con datos de prueba y prediccion
        Salida: exactitud, precision y recall          
        """ 
        #r contiene los indices donde la prediccion y los valores reales coinciden
        r = np.where(y_predicted==y_test)
        acc = y_test[r]
        
        y_p = np.where(y_predicted>0)
        pos=np.where(y_test>0)
        
        FP = set(list(y_p[0])).symmetric_difference(set(list(pos[0])))
        TP = set(list(pos[0]))-FP        
        TP_FP = TP.union(FP) 
                
        #exactitud, precision y sensibilidad
        accuracy= len(acc)/len(y_test)*100
        precision = len(list(TP))/len(list(TP_FP))*100
        recall = len(list(TP))/len(pos[0])*100
        
        return accuracy, precision, recall    
    
def test_basico(X_train,y_train,X_test,y_test):    
    #Declara el modelo
    clases = np.unique(y_train).astype(int)
    clf=modelo_basico(clases)
    
    #Entrenamiento
    x,y=clf.fit(X_train,y_train)

    #frontera de decision
    ax.plot([x[0],y[0]],[x[1],y[1]],color='olive', label='Basico')

    #PRUEBA
    y_p=clf.predict(X_test)
    y_p = np.where(y_p<0,clases[0],clases[1])
    ex,pr = clf.metricas(y_test,y_p)
    print('MODELO BASICO:')
    print("Exactitud   " + str(ex) + "%")
    print("Precision   " + str(pr) + "%")
    return

