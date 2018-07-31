"""
2次元型Talbot型X線位相イメージング装置で検出される干渉像の簡易シミュレータ
簡単化のため干渉格子やX線の伝搬等はシミュレートせず理論的に予測される像を
最低限のパラメータで直接計算する

2018/7/29 K. Nagai
"""

import numpy as np
import scipy as sp
import matplotlib.pylab as plt
pi = np.pi
e = np.e

n=128
l_micron=12800 #CCD素子の大きさは100umを仮定
period=4
fac=0.5
keV = 22.0
xlambda = 12.40/keV/10000.0
cdds = l_micron/n
m_photon = 2000
i = complex(0,1)


def CalcObject(n, xlambda, l_micron):
    """
    オブジェクトのX線透過時の位相差及び吸収量を簡易的に計算
    今回は簡略化のため、ポリスチレンが22keVのX線を透過する場合のみ
    """
    
    
    # メッシュの作成
    x = [x0/n*l_micron-(l_micron/2) for x0 in range(n)]
    y = [y0/n*l_micron-(l_micron/2) for y0 in range(n)]    
    xx, yy = np.meshgrid(x,y)
 
    OBJ = np.zeros((n,n))
    
    # 22.0keVのX線に対するポリスチレン複素屈折率
    # 以下のサイトで計算
    # http://purple.ipmt-hpm.ac.ru/xcalc/xcalc_mysql/ref_index.php
    ObjDelta = 4.613501188287E-07 
    ObjBeta = 6.140380620628E-11
    
    # 球の半径
    rad = 3000
    
    # X線が生じる球の位相差の計算
    ObjPhase = 2.0*pi*i/xlambda*(-ObjDelta+i*ObjBeta)
    OBJ = np.where((xx**2.0+yy**2.0)<rad**2.0, sp.sqrt(rad**2.0-(xx**2.0+yy**2.0))*ObjPhase*2.0, np.zeros((n,n)))
    return OBJ

def SPFunc(n, l_micron, sw):
    """
    Point Spread Functionの計算
    検出器や装置の攻勢によるボケ量を計算する
    画像へのボケの計算はコンボリューションを使用する
    """
    
    x = [x0/n*l_micron-(l_micron/2) for x0 in range(n)]
    y = [y0/n*l_micron-(l_micron/2) for y0 in range(n)]    
    xx, yy = np.meshgrid(x,y)
    
    SS = np.exp(-(xx**2+yy**2)/2/sw**2)
    Sum = np.sum(SS)
    SS = np.fft.fftshift(SS)
    return SS/Sum
      
# メッシュグリッド
x = [x0 for x0 in range(n)]
y = [y0 for y0 in range(n)]    
xx, yy = np.meshgrid(x,y)

# オブジェクトの位相の計算
OBJ = CalcObject(n, xlambda, l_micron)

Phs = np.imag(OBJ)*4.0/2.0*3.0 # 第三タルボ位置のPhase変化量計算
Abs = np.exp(-np.real(OBJ)) # 吸収係数を分離

SPF = SPFunc(n, l_micron, 50.0) # 装置構成による光源ボケをConvolutionで表現
Phs = np.real(np.fft.ifft2(np.fft.fft2(Phs)*np.fft.fft2(SPF)))
Abs = np.real(np.fft.ifft2(np.fft.fft2(Abs)*np.fft.fft2(SPF)))

DPx = np.diff(Phs, axis = 1)/cdds # 微分位相 x
DPx = np.hstack((DPx, np.transpose(np.array([DPx[:,0]]))))
DPy = np.diff(Phs, axis = 0)/cdds # 微分位相 y
DPy = np.vstack((DPy, np.array(DPy[0,:])))

# 取得される干渉像(モアレ)の簡易計算
Moire = Abs*(1.0-0.5*np.cos(xx*2.0*pi/period-DPx))*(1.0-0.5*np.cos(yy*2.0*pi/period-DPy))/4.0
# Ref = (1.0-0.5*np.cos(xx*2.0*pi/period))*(1.0-0.5*np.cos(yy*2.0*pi/period))/4.0 # 被写体なし

SPF = SPFunc(n, l_micron, 50.0) # 検出器によるボケをConvolutionで表現
Moire = np.real(np.fft.ifft2(np.fft.fft2(Moire)*np.fft.fft2(SPF)))
np.save('Moire.npy', Moire)

plt.figure(figsize=(12,11))
plt.imshow(Moire, cmap='bone')
plt.savefig('MoireImg.png')    