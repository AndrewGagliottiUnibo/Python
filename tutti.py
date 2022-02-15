import numpy as np;
import scipy.linalg as spl;
import math;
import sympy as sym;
import scipy.optimize as op
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.fft import fftshift
from scipy.fftpack import ifft
from scipy.fftpack import ifftshift

# FUNZIONI ZERI #
def errore_relativo(valesatto,valcalcolato):
  if valcalcolato==0:return abs(valesatto)
  return abs((valcalcolato-valesatto)/valcalcolato)
def sign(val):return math.copysign(1,val)
def stima_ordine(X):
  k=len(X)-4
  if k<0:print("stima non calcolabile");return None;
  return (np.log(abs(X[k+2]-X[k+3])/abs(X[k+1]-X[k+2]))
          / np.log(abs(X[k+1]-X[k+2])/abs(X[k]-X[k+1])))
def calcolo_ordine(alfa,tol,Iterable,expression,modules):
  p,fxp,y,C=0,expression,0,1
  while abs(y)<=tol:
    p+=1
    fx,fxp=fxp,sym.diff(fxp,Iterable,1)
    if fx==fxp:
      print("la derivata di grado",p,"è uguale alla derivata di grado",p-1, \
              "(",fx,"=",fxp,") interrompo il ciclo")
      return 0,0
    fname=sym.lambdify(Iterable,fxp,modules)
    y,C=fname(alfa),C*p
  return p,abs(y)/C
def bisez(f,a,b,tol):
  if a>b:b,a=a,b
  ya,yb=f(a),f(b)
  if sign(ya)==sign(yb):print("segno concorde agli estremi");return None,0,[]
  m,X=None,[]
  while abs(b-a)>=tol:
    m=a+(b-a)/2#algoritmo stabile punto medio a-b
    if a==m or b==m:break#spacing raggiunto
    ym=f(m);X.append(m)
    if ym==0:break#trovato zero
    if sign(ya)==sign(ym):
      a,ya=m,ym
    else:
      b,yb=m,ym
  return m,len(X),X
def regula_falsi(f,a,b,tol,maxit):
  if tol==None or tol<0:tol=0
  if a>b:b,a=a,b
  ya,yb=f(a),f(b)
  if sign(ya)==sign(yb):print("segno concorde agli estremi");return None,0,[]
  ym,m,X=ya,None,[]
  while len(X)<maxit and not(abs(ym)<tol and abs(b-a)<tol):
    m=a+ya*(b-a)/(ya-yb)#intersezione con asse x del segmento ya-yb
    if a==m or b==m:break;#spacing raggiunto
    ym=f(m);X.append(m)
    if ym==0:break#trovato zero
    if sign(ya)==sign(ym):
      a,ya=m,ym
    else:
      b,yb=m,ym
  return m,len(X),X
def corde(f,f1,x0,tolx,toly,maxit):
  xk=[]
  m=f1(x0)#m=Coefficiente angolare della tangente in x0
  if m==0:print("derivata prima nulla all'innesco");return None,0,[]
  while True:
    fx0=f(x0);d=fx0/m;x1=x0-d;fx1=f(x1);xk.append(x1)
    if len(xk)>=maxit or abs(fx1)<toly or abs(d)<tolx*abs(x1):break
    x0=x1
  return x1,len(xk),xk
def secanti(fname,xm1,x0,tolx,tolf,nmax):
  xk=[]
  while True:
    fxm1,fx0=fname(xm1),fname(x0)
    d=fx0*(x0-xm1)/(fx0-fxm1);x1=x0-d;fx1=fname(x1);xk.append(x1)
    if len(xk)>=nmax or abs(fx1)<tolf or abs(d)<tolx*abs(x1):break
    xm1,x0=x0,x1
  return x1,len(xk),xk

def newton(fname,fpname,x0,tolx,tolf,nmax):
  eps,xk,x1=np.spacing(1),[],None
  while True:
    fx0,dfx0=fname(x0),fpname(x0)
    if abs(dfx0)>eps:
      d=fx0/dfx0;x1=x0-d;fx1=fname(x1);xk.append(x1)
    else:
      print("Derivata nulla in x0 Newton, uscita anticipata")
      return x1,len(xk),xk
    if len(xk)>=nmax or abs(fx1)<tolf or abs(d)<tolx*abs(x1):break;
    x0=x1
  return x1,len(xk),xk
def newton_m(fname,fpname,x0,m,tolx,toly,nmax):
  eps,xk,x1=np.spacing(1),[],None
  while True:
    fx0=fname(x0);dfx0=fpname(x0)
    if abs(dfx0)>eps:
      x1=x0-m*fx0/dfx0;fx1=fname(x1);xk.append(x1)
    else:
      print("Newton modificato: derivata nulla in",x0); return x1,len(xk),xk
    if len(xk)>=nmax or abs(fx1)<toly or abs(x1-x0)<tolx:break
    x0=x1
  return x1,len(xk),xk
def iterazione(gname,x0,tolx,nmax):
  xk=[];xk.append(x0)
  while True:
    x1=gname(x0);d=x1-x0;xk.append(x1)
    if len(xk)>=nmax or abs(d)<tolx*abs(x1):break
    x0=x1
  return x1,len(xk),xk
def Lsolve(L,b):
  if len(L.shape)!=2:raise ValueError("Lsolve: Matrice non 2 dimensioni..",L)
  n,m=L.shape
  if n!=m:raise ValueError("Lsolve: Matrice non quadrata.",L)
  if not np.all(np.diag(L)):raise ValueError("Lsolve: El. diagonale 0.",L)
  x=np.zeros((n,))
  for i in range(n):x[i]=(b[i]-np.dot(L[i,:i],x[:i]))/L[i,i]
  return x
# FUNZIONI SISTEMI LINEARI #
def Usolve(U,b):
  if len(U.shape)!=2:raise ValueError("Usolve: Matrice non 2 dimensioni.",U)
  n,m=U.shape
  if n!=m:raise ValueError("Usolve: Matrice non quadrata.",U)
  if not np.all(np.diag(U)):raise ValueError("Usolve: El. diagonale 0.",U)
  x=np.zeros((n,))
  for i in range(n-1,-1,-1):x[i]=(b[i]-np.dot(U[i,i+1:n],x[i+1:n]))/U[i,i]
  return x
def LUsolve(L,U,P,b):
  return Usolve(U,Lsolve(L,np.dot(P,b)))#PA=LU,Ax=b->PAx=Pb->L(Ux)=Pb
def LU_nopivot(A):
  if len(A.shape)!=2:raise ValueError("LU_nopivot: Matr. non 2 dimensioni.",A)
  n,m,U=A.shape,A.copy()
  if n!=m: raise ValueError("LU_nopivot: Matrice non quadrata.",A)
  for k in range(n-1):
    if U[k,k]==0:
      raise ValueError('LU_nopivot: El. diagonale 0 al passo '+str(k)+'.',U)
    U[k+1:n,k]=U[k+1:n,k]/U[k,k]
    U[k+1:n,k+1:n]=U[k+1:n,k+1:n]-np.outer(U[k+1:n,k],U[k,k+1:n])
  return np.tril(U,-1)+np.eye(n),np.triu(U)
def swapRows(A,k,p):A[[k,p],:]=A[[p,k],:]
def LU_pivot(A):
  if len(A.shape)!=2:raise ValueError("LU_pivot: Matrice non 2 dimensioni.",A)
  n,m=A.shape
  if n!=m:raise ValueError("LU_pivot: Matrice non quadrata.",A)
  P,U=np.eye(n),A.copy()
  for k in range(n-1):
    p=np.argmax(abs(U[k:n,k]))+k
    if p!=k:
      swapRows(P,k,p);swapRows(U,k,p)
    U[k+1:n,k]/=U[k,k]
    U[k+1:n,k+1:n]-=np.outer(U[k+1:n,k],U[k,k+1:n])
  return P,np.tril(U,-1)+np.eye(n),np.triu(U)
def solve_nopivot(A,b):
  if b.shape[0]!=A.shape[1]:
    raise ValueError("solve_nopivot: Vett. dei termini noti incongruo.",A,b)
  L,U=LU_nopivot(A)
  return Usolve(U,Lsolve(L,b))
def solve_pivot(A,b):
  if b.shape[0]!=A.shape[0]:
    raise ValueError("solve_pivot: Vettore termini non compatibile.",A,b)
  P,L,U=LU_pivot(A)
  return LUsolve(L,U,P,b);
def solve(A,b,pivot):
  if pivot:
    P,L,U=LU_pivot(A)
  else:
    L,U=LU_nopivot(A)
    P=np.eye(A.range[0])
  return LUsolve(L,U,P,b)
def solve_nsis(A,B):
  m,n=A.shape
  if n!=B.shape[1]:raise ValueError("solve_nsis: Matrici non congruenti.",A,B)
  X=np.zeros((n,n))
  P,L,U=LU_nopivot(A)
  for i in range(n):
    X[:,i]=Usolve(U,Lsolve(L,np.dot(P,B[:,i])))
  return X
# FUNZIONI ITERPOLAZIONE #
def nodi_cheb(a,b,n):#Nodi di chebyshev
    Cheb=lambda i:((a+b)/2)+((b-a)/2)*np.cos(((2*i+1)/(2*(n+1))*math.pi))
    return Cheb(np.arange(0,n+1,dtype=float))
    
def plagr(xnodi,k):
  xzeri,n=np.zeros_like(xnodi),xnodi.size
  xzeri=(xnodi[1:n] if k==0 else np.append(xnodi[0:k],xnodi[k+1:n]))
  num=np.poly(xzeri)
  den=np.polyval(num,xnodi[k])
  return num/den
def InterpL(x,f,xx):
  L=np.zeros((x.size,xx.size))
  for k in range(x.size):L[k,:]=np.polyval(plagr(x,k),xx)
  return np.dot(f,L) 
def zeri_Cheb(a,b,n):
  if a>b:a,b=b,a
  t1,t2,x=(a+b)/2,(b-a)/2,np.zeros((n+1,))
  for k in range(n+1):x[k]=t1+t2*np.cos(((2*k+1)/(2*(n+1))*np.pi))
  return x
# FUNZIONI INTEGRAZIONE #
def TrapComp(fname,a,b,n):
  h=(b-a)/n;f=fname(np.arange(a,b+h,h))
  return (f[0]+2*np.sum(f[1:n])+f[n])*h/2
def SimpComp(fname,a,b,n):
  h=(b-a)/(2*n);f=fname(np.arange(a,b+h,h))
  I=(f[0]+2*np.sum(f[2:2*n:2])+4*np.sum(f[1:2*n:2])+f[2*n])*h/3
  return I
def traptoll(fun,a,b,tol):
  Nmax,err,N=2048,1,1;IN=TrapComp(fun,a,b,N);
  while err>tol:
    N=2*N
    if N>Nmax:print('traptoll: Raggiunto n. MAX iterazioni.');return [],0
    I2N=TrapComp(fun,a,b,N);err=abs(IN-I2N)/3;IN=I2N
  return IN,N
def simptoll(fun,a,b,tol):
  Nmax,err,N=2048,1,1;IN=SimpComp(fun,a,b,N);
  while err>tol:
    N=2*N
    if N>Nmax:print('simptoll: Raggiunto n. MAX iterazioni.');return [],0
    I2N=SimpComp(fun,a,b,N);err=abs(IN-I2N)/15;IN=I2N
  return IN,N
# FUNZIONI MINIMI QUADRATI #
def metodoQR(x,y,n):
  Q,R=spl.qr(np.vander(x,n+1))
  return Usolve(R[0:n+1,:],np.dot(Q.T,y))
def minimi_quadrati():
	x=np.arange(1900,2011,10)
	y=np.array([76.0,92.0,106.0,123.0,132.0,151.0,179.0,203.0,226.0,249.0,281.0,305.0])
	xmin=np.min(x)
	xmax=np.max(x)
	xval=np.linspace(xmin,xmax,100)
	legends=[]
	for n in range(1,8):
	    a=metodoQR(x,y,n)
	    px=np.polyval(a,x)
	    residuo=np.sum((y-px)**2)
	    print("Norma del residuo al quadrato grado {0}".format(n),residuo)
	    legends.append("n={0}, residuo={1}".format(n,residuo))
	    p=np.polyval(a,xval)
	    plt.plot(xval,p)
	legends.append("punti")
	plt.plot(x,y,'o')
	plt.legend(legends)

def fourier():
    xt1=lambda x:np.sin(2*np.pi*15*x)*4
    xt2=lambda x:np.sin(2*np.pi*40*x)*3
    xt3=lambda x:np.sin(2*np.pi*60*x)*2
    xt=lambda x:xt1(x)+xt2(x)+xt3(x)
    xp=lambda x:np.sin(2*np.pi*80*x)*2
    xtp=lambda x:xt(x)+xp(x)
    T=2
    FreqCamp=170
    NrCampioni=T*FreqCamp
    PassoCampionamento=1/FreqCamp
    X=np.linspace(0,T,NrCampioni)
    Yt=xt(X)
    Yp=xp(X)
    Ytp=xtp(X)
    plt.title("Segnale originale")
    plt.plot(X,Yt)
    plt.show()
    plt.title("disturbo")
    plt.plot(X,Yp)
    plt.show()
    plt.title("segnale disturbato")
    plt.plot(X,Ytp)
    plt.show()
    freqFourier=np.linspace(-FreqCamp/2,FreqCamp/2,NrCampioni,endpoint=True)
    Fourier=fftshift(fft(Ytp))
    plt.plot(freqFourier, Fourier)
    #il rumore ha 80hz, il segnale pulito ha massimo 60Hz, posso filtrare a 75
    FrequenzaFiltro=75
    
    FreqDaTogliere=np.abs(freqFourier)>FrequenzaFiltro
    FourierPulito=np.copy(Fourier)
    FourierPulito[FreqDaTogliere]=0
    YRicostruito=ifft(ifftshift(FourierPulito))
    plt.plot(freqFourier, FourierPulito)
    plt.legend(["Spettro fourier Segnale rumoroso","Spettro fourier SegnalePulito"])
    plt.show()
    plt.plot(X, Yt)
    plt.plot(X, YRicostruito)
    plt.legend(["Segnale originale","Segnale Pulito"])
    plt.show()
    return("fine es2_parte2")
    


    
def ClearConsole():
    print(chr(27)+"[J")
    
np.arange(1,2).all()