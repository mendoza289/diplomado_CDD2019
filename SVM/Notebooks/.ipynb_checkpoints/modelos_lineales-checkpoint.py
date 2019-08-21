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
import matplotlib as mpl
import matplotlib.pyplot as plt
import genera_datos as gd


def plot_linea(centro, wT, xmin, xmax):
    """
    Calcula los puntos de una linea recta 
    cuya direccion esta dada por un vector dir
    """
    num_p=len(np.arange(0,np.abs(xmax-xmin),0.5))
    dc_p =  num_p // 2
    dc_n =  -num_p // 2
    negp = centro + dc_n * wT
    posp = centro + dc_p * wT
    return negp, posp

def f(x, w, b, c=0):
    # dados x y w, regresa y tal que [x,y] esta sobre la linea
    # w.x + b = c
    #w es el vector ortogonal a la recta
    v=(-w[0] * x - b + c) / w[1]
    return v

def plot_margin(X1_train, X2_train, clf):

    x=np.vstack((X1_train[:,0],X2_train[:,0]))
    xmin = np.amin(x)+0.1*np.amin(x) 
    xmax = np.amax(x)+0.1*np.amax(x) 

    a0 = xmin
    a1 = f(a0, clf.w, clf.b)

    b0 = xmax 
    b1 = f(b0, clf.w, clf.b)

    return [a0,b0],[a1,b1]


def plot_contour(X1_train, X2_train, clf):
    #vectores de soporte
    #plt.scatter(clf.sv[:,0], clf.sv[:,1], s=100, facecolor='none',edgecolor="g",linewidth='3')
    
    x=np.vstack((X1_train[:,0],X2_train[:,0]))
    y=np.vstack((X1_train[:,1],X2_train[:,1]))
    xmin = np.amin(x)+0.1*np.amin(x) 
    xmax = np.amax(x)+0.1*np.amax(x) 
    ymin = np.amin(y)+0.1*np.amin(y) 
    ymax = np.amax(y)+0.1*np.amax(y)
    plt.axis([xmin,xmax,ymin,ymax])
    
    X1, X2 = np.meshgrid(np.linspace(xmin,xmax,20), np.linspace(ymin,ymax,20))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = clf.project(X).reshape(X1.shape)
    plt.contour(X1, X2, Z, [0.0], colors='m', linewidths=1,origin='lower',label='Perceptron Kernel')

    plt.axis("tight")
    plt.show()

def datos_lin_separables(clusters=2,samples=100):
    """
    Genera datos linealmente separables
    Se generan hasta 10 clusters distintos
    Entrada: numero de clusters y numero de muestras
    Salida: Lista con matrices de vectores de datos por 
    clase.
    """
    if clusters > 10:
        n = 10
        print('solo se pueden producir hasta 10 clusters')
    elif clusters < 2:
        n = 2
        print('el numero minimo de clusetrs es 2')
    else:
        n = clusters
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
    
    rho = np.random.uniform(low=0.2, high=0.8, size=(n,))
    if n==2:
        p=np.random.rand()
        if p < 0.5:
            rho[0]=-rho[0]
            rho[1]=-rho[1]
    if n > 3:
        for i in range(n):
            rho[i] *= (-1)**i
    
    s = np.random.rand(2*n)+1
    s = s.reshape((n,2))
    
    cov_list = []
    for i,v in enumerate(s):
        cov = np.zeros((2,2))
        cov[0,0] = v[0]**2
        cov[1,1] = v[1]**2
        cov[0,1] = v[0]*v[1]*rho[i]
        cov[1,0] = cov[0,1]
        cov_list.append(cov)
    
    L = []
    
    clases = np.arange(n,dtype='int')
    if n==2:
        clases[0]=-1
        
    y = np.repeat(clases[0],samples).reshape(samples,1).astype(int)
    
    for i in range(1,n):
        yp = np.repeat(clases[i],samples).reshape(samples,1).astype(int)
        y = np.hstack((y,yp))
    
    for i in range(n):
        X = np.random.multivariate_normal(mus[i], cov_list[i],samples)
        X = np.hstack((X,y[:,i].reshape(samples,1).astype(int)))
        L.append(X)
        
    return np.array(L)
  
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, gamma):
    return np.exp(-gamma*np.linalg.norm(x-y)**2)


class modelo_LSTSQ(object):
    """
    Modelo basico de clasificacion binaria
    Entrada: datos de entrenamiento X,y
    Salida: valores de la frontera de decision
    """
    def __init__(self):
        self.w = np.zeros(2)
    
    def fit(self, X_train,y_train):
        #modelo basico
        m_B1=np.array([X_train[:,0][y_train<0].mean(),X_train[:,1][y_train<0].mean()])
        m_B2=np.array([X_train[:,0][y_train>0].mean(),X_train[:,1][y_train>0].mean()])
        
        #centroides de los datos
        plt.scatter([m_B2[0]],[m_B2[1]],facecolor='r',edgecolor='k')
        plt.scatter([m_B1[0]],[m_B1[1]],facecolor='k',edgecolor='k')
        plt.plot([m_B1[0],m_B2[0]],[m_B1[1],m_B2[1]],color='olive',ls='--',lw=1)
        plt.show()
        
        Xs=gd.inv_(gd.scatter(X_train))

        self.w=np.dot(Xs,(m_B2-m_B1))
    
        wT = np.array([-self.w[1],self.w[0]])
        #dir_90=np.array([dir[0],-dir[1]])
    
        medioX=m_B1[0]/2+m_B2[0]/2
        medioY=m_B1[1]/2+m_B2[1]/2
        centro=np.array([medioX,medioY])
                
        xmin = np.amin(X_train[:,0])+0.1*np.amin(X_train[:,0]) 
        xmax = np.amax(X_train[:,0])+0.1*np.amax(X_train[:,0]) 
        self.x,self.y=plot_linea(centro, wT, xmin, xmax)

        return self.x, self.y
    
    def predict(self,X):
        return np.sign(np.dot(X,self.w))

    def metricas(self, y_t,y_test):
        r=np.where(y_t==y_test)
        y_p=np.where(y_t>0)
        rp=y_test[y_p]
        rp_t=rp[rp>0]
        exact= len(r[0])/len(y_test)*100
        prec = len(rp_t)/len(rp)*100
        return exact, prec

class modelo_basico(object):
    """
    Modelo basico de clasificacion binaria
    Entrada: datos de entrenamiento X,y
    Salida: valores de la frontera de decision
    """
    def __init__(self):
        self.w = np.zeros(2)
    
    def fit(self, X_train,y_train):
        #modelo basico
        m_B1=np.array([X_train[:,0][y_train<0].mean(),X_train[:,1][y_train<0].mean()])
        m_B2=np.array([X_train[:,0][y_train>0].mean(),X_train[:,1][y_train>0].mean()])
        
        #centroides de los datos
        plt.scatter([m_B2[0]],[m_B2[1]],facecolor='r',edgecolor='k')
        plt.scatter([m_B1[0]],[m_B1[1]],facecolor='k',edgecolor='k')
        plt.plot([m_B1[0],m_B2[0]],[m_B1[1],m_B2[1]],color='olive',ls='--',lw=1)
        plt.show()

        self.w=m_B2-m_B1
    
        wT = np.array([-self.w[1],self.w[0]])
        #dir_90=np.array([dir[0],-dir[1]])
    
        medioX=m_B1[0]/2+m_B2[0]/2
        medioY=m_B1[1]/2+m_B2[1]/2
        centro=np.array([medioX,medioY])
        
        self.t=np.dot(self.w,centro)
        
        xmin = np.amin(X_train[:,0])+0.1*np.amin(X_train[:,0]) 
        xmax = np.amax(X_train[:,0])+0.1*np.amax(X_train[:,0]) 
        self.x,self.y=plot_linea(centro, wT, xmin, xmax)

        return self.x, self.y
    
    def predict(self,X):
        return np.sign(np.dot(X,self.w))

    def metricas(self, y_t,y_test):
        r=np.where(y_t==y_test)
        y_p=np.where(y_t>0)
        rp=y_test[y_p]
        rp_t=rp[rp>0]
        exact= len(r[0])/len(y_test)*100
        prec = len(rp_t)/len(rp)*100
        return exact, prec



class Perceptron(object):

    def __init__(self, T=1, eta=1.):
        self.T = T
        self.eta=eta

    def fit(self, X, y):
        n_samples, d_features = X.shape
        self.w = np.zeros(d_features, dtype=np.float64)
        self.b = 0.0

        for t in range(self.T):
            for i in range(n_samples):
                if self.predict(X[i]) != y[i]:
                    self.w += y[i] * self.eta * X[i]
                    self.b += self.eta*y[i]
        
    def project(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.sign(self.project(X))
    
    def metricas(self, y_t,y_test):
        r=np.where(y_t==y_test)
        y_p=np.where(y_t>0)
        rp=y_test[y_p]
        rp_t=rp[rp>0]
        exact= len(r[0])/len(y_test)*100
        prec = len(rp_t)/len(rp)*100
        return exact, prec

class KernelPerceptron(object):

    def __init__(self, kernel=linear_kernel, T=1,gamma_p=1.0):
        self.kernel = kernel
        self.T = T
        self.gamma=gamma_p
        self.p=gamma_p
        if self.kernel==linear_kernel:
            print('KERNEL LINEAL')
        elif self.kernel==gaussian_kernel:
            print('KERNEL GAUSSIANO; gamma={}'.format(self.gamma))
        else:
            print('KERNEL POLINOMIAL; p={}'.format(self.p))
            
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples, dtype=np.float64)
        
        for j in range(self.T):
            for i in range(n_samples):
                xi=X[i]
                yi=y[i]
                if self.kernel==gaussian_kernel:    
                    if yi*np.sum(self.alpha*y*self.kernel(X,xi,self.gamma)) <= 0:
                        self.alpha[i] += 1.0
                elif self.kernel==polynomial_kernel:    
                    if yi*np.sum(self.alpha*y*self.kernel(X,xi,self.p)) <= 0:
                        self.alpha[i] += 1.0
                else:
                    if yi*np.sum(self.alpha*y*self.kernel(X,xi)) <= 0:
                        self.alpha[i] += 1.0
        
        # Support vectors
        sv=np.argwhere(self.alpha>0.).ravel()
        print('support vectors @:',sv)
        self.alpha = self.alpha[sv].ravel()
        self.sv=X[sv]
        self.ysv=y[sv]
        print("{} vectores de soporte de {} puntos".format(len(self.sv),n_samples))

    def project(self,X):
        n_samples, n_features = X.shape
        y_predict = np.zeros(len(X))
        for i in range(n_samples):
            s=0
            xi=X[i]
            for j in range(len(self.sv)):
                if self.kernel==gaussian_kernel:    
                    s += self.alpha[j]*self.ysv[j]*self.kernel(xi,self.sv[j],self.gamma)
                elif self.kernel==polynomial_kernel:    
                    s += self.alpha[j]*self.ysv[j]*self.kernel(xi,self.sv[j],self.p)
                else:
                    s += self.alpha[j]*self.ysv[j]*self.kernel(xi,self.sv[j])
            y_predict[i]=s
        return y_predict
        
    def predict(self, X):
        return np.sign(self.project(X))

    def metricas(self, y_t,y_test):
        r=np.where(y_t==y_test)
        y_p=np.where(y_t>0)
        rp=y_test[y_p]
        rp_t=rp[rp>0]
        exact= len(r[0])/len(y_test)*100
        prec = len(rp_t)/len(rp)*100
        return exact, prec

if __name__ == "__main__":
    
    def test_basico(X_train,y_train,X_test,y_test):    
        #plt.savefig(my_path+'datos_separables.png',bbox_inches='tight')
        clf=modelo_basico()
        x,y=clf.fit(X_train,y_train)
        xmin = np.amin(X_train[:,0])+0.1*np.amin(X_train[:,0]) 
        xmax = np.amax(X_train[:,0])+0.1*np.amax(X_train[:,0]) 
        ymin = np.amin(X_train[:,1])+0.1*np.amin(X_train[:,1]) 
        ymax = np.amax(X_train[:,1])+0.1*np.amax(X_train[:,1]) 
        plt.axis([xmin,xmax,ymin,ymax])
        plt.plot([x[0],y[0]],[x[1],y[1]],color='olive',label='Basico')
    
        #PRUEBA
        y_t=clf.predict(X_test)
        ex,pr = clf.metricas(y_t,y_test)
        
        plt.scatter(X_test[:,0][y_test>0], X_test[:,1][y_test>0],facecolor='r', marker='o', edgecolor='k',s=45)
        plt.scatter(X_test[:,0][y_test<0], X_test[:,1][y_test<0],facecolor='k', marker='o', edgecolor='k',s=45)

        print('MODELO BASICO:')
        print("Exactitud   " + str(ex) + "%")
        print("Precision   " + str(pr) + "%")

        plt.show()
        
    def test_perceptron(X_train,y_train,X_test,y_test): 
        #ENTRENA
        clf = Perceptron(T=20)
        clf.fit(X_train, y_train)
        
        x,y=plot_margin(X_train[y_train>0], X_train[y_train<=0], clf)
        ymin,ymax=plt.ylim()
        plt.axis([x[0],x[1],ymin,ymax])
        plt.plot(x, y, "k",label='Perceptron DUAL')
        
        #PRUEBA
        y_t=clf.predict(X_test)
        ex,pr = clf.metricas(y_t,y_test)
        print('PERCEPTRON DUAL:')
        print("Exactitud   " + str(ex) + "%")
        print("Precision   " + str(pr) + "%")

        plt.show()
        
    def test_kernel(X_train,y_train,X_test,y_test):
        clf = KernelPerceptron(gaussian_kernel,T=10,gamma_p=0.2925)
        #clf = KernelPerceptron(polynomial_kernel,T=10,gamma_p=3)
        clf.fit(X_train, y_train)
        
        #PRUEBA
        y_t=clf.predict(X_test)
        ex,pr = clf.metricas(y_t,y_test)
        print('PERCEPTRON KERNEL:')
        print("Exactitud   " + str(ex) + "%")
        print("Precision   " + str(pr) + "%")

        plot_contour(X_train[y_train<0], X_train[y_train>0], clf)
        #plt.savefig(my_path+'modelos_lineales_kernel_gaus.png',bbox_inches='tight')
        return

    def test_LSTSQ(X_train,y_train,X_test,y_test):    
        clf=modelo_LSTSQ()
        x,y=clf.fit(X_train,y_train)
        xmin = np.amin(X_train[:,0])+0.1*np.amin(X_train[:,0]) 
        xmax = np.amax(X_train[:,0])+0.1*np.amax(X_train[:,0]) 
        ymin = np.amin(X_train[:,1])+0.1*np.amin(X_train[:,1]) 
        ymax = np.amax(X_train[:,1])+0.1*np.amax(X_train[:,1]) 
        plt.axis([xmin,xmax,ymin,ymax])
        plt.plot([x[0],y[0]],[x[1],y[1]],color='m',label='LSTSQ')
    
        #PRUEBA
        y_t=clf.predict(X_test)
        ex,pr = clf.metricas(y_t,y_test)
        
        plt.scatter(X_test[:,0][y_test>0], X_test[:,1][y_test>0],facecolor='r', marker='o', edgecolor='k',s=45,label='pos test')
        plt.scatter(X_test[:,0][y_test<0], X_test[:,1][y_test<0],facecolor='k', marker='o', edgecolor='k',s=45,label='neg test')

        print('MODELO LSTSQ:')
        print("Exactitud   " + str(ex) + "%")
        print("Precision   " + str(pr) + "%")

        plt.savefig(my_path+'modelos_lineales_1.png',bbox_inches='tight')
        legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=3, fancybox=True, shadow=False)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_alpha(1)
        plt.show()
    
    
    my_path='/Users/jorge/Google Drive/Clases/MACHINE LEARNING/Sesiones/Imagenes/'
    X1,y1,X2,y2=gd.datos_lin_separables(100,0.52)
    #X1,y1,X2,y2=gd.datos_solapados(150)
    #X1,y1,X2,y2=datos_no_lin_separables(100)
    
    X_train,y_train,X_test,y_test=gd.split_train_test(X1,y1,X2,y2,80)        
    print('|X_train|= {} ; |X_test|= {}'.format(len(X_train),len(X_test)))
    
    X_train=gd.centra_datos(X_train)
    X_test=gd.centra_datos(X_test)
    #X_train=normaliza(X_train)
    #X_test=normaliza(X_test)
    
    plt.scatter(X_train[:,0][y_train>0], X_train[:,1][y_train>0],facecolor='orangered', marker='$\\bigoplus$', edgecolor='none',s=90,label='pos train')
    plt.scatter(X_train[:,0][y_train<0], X_train[:,1][y_train<0],facecolor='royalblue', marker='$\\ominus$', edgecolor='none',s=90,label='neg train')
    
    plt.show()
    test_basico(X_train,y_train,X_test,y_test)
    test_perceptron(X_train,y_train,X_test,y_test)
    #test_kernel(X_train,y_train,X_test,y_test)
    test_LSTSQ(X_train,y_train,X_test,y_test)
