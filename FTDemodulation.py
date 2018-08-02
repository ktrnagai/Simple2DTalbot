"""
2次元型Talbot型X線位相イメージング装置で検出されたモアレ像から
フーリエ変換法を用いて元の被写体の情報を再現する
取得できる情報は
被写体に吸収されたX線像（旧来のX線像）
被写体にX線が透過した際の屈折率のシア像(x方向、y方向)
被写体に小角散乱されたX線の情報(x方向,y方向)

2018/7/29 K. Nagai
"""
import numpy as np
import scipy as sp
import matplotlib.pylab as plt
pi = np.pi
e = np.e
n=128
period=4
i = complex(0,1) #虚数単位
plt.figure(figsize=(12,11))


def FFTDemod(Moire):
    """
    フーリエ変換法による干渉縞から被写体情報を取得する関数
    """
    
    # フーリエ変換実行
    FTM = np.fft.fftshift(np.fft.fft2(Moire))    
    FTimage = np.log10(np.abs(FTM))
    plt.imshow(FTimage, cmap='bone')
    plt.savefig('FTimage.png')    
    plt.clf()
    

    # 切り取り関数を計算(ハン窓)
    x = [x0-n/2 for x0 in range(n)]
    y = [y0-n/2 for y0 in range(n)]    
    xx, yy = np.meshgrid(x,y)
    rad = n/4 # 4画素周期のため
    HanW = np.where((xx**2+yy**2)<rad**2, 1+np.cos(sp.sqrt(xx**2+yy**2)/rad*pi), np.zeros((n,n)))
    
    # 領域切り出し
    AB = np.fft.ifft2(np.fft.ifftshift(FTM*HanW))
    PX = np.fft.ifft2(np.fft.ifftshift(np.roll(FTM, int(rad), axis=1)*HanW))
    PY = np.fft.ifft2(np.fft.ifftshift(np.roll(FTM, int(rad), axis=0)*HanW))
    
    # 情報取り出し
    ABS = 1-np.abs(AB)
    DPx = -np.angle(PX*np.exp(i*pi))
    SPx = np.abs(PX)
    DPy = -np.angle(PY*np.exp(i*pi))
    SPy = np.abs(PY)
        
    return ABS, DPx, DPy, SPx, SPy

def Integration(PDx, PDy):
    """
    微分情報(dI/dx,dI/dy)からもとの位相Iを積分して求める
    フーリエ関数による積分を利用
    """
    # フーリエ変換とコンボリューションを用いた積分    
    A= PDx + i*PDy
    
    # 境界条件のために反転させた行列を接続する
    A2 = np.hstack((A, np.fliplr(A)))
    A3 = np.vstack((A2, np.flipud(A2)))
    
    FA = np.fft.fft2(A3)
    
    # 積分用の演算子作成
    kx = [(kx0-n)/n for kx0 in range(n*2)]
    ky = [(ky0-n)/n for ky0 in range(n*2)]
    kxx, kyy = np.meshgrid(kx,ky)
    
    with np.errstate(divide = "ignore", invalid="ignore"):
        OP = 1.0/(pi*i*(kxx+i*kyy))
        OP = np.fft.ifftshift(OP)
        OP[np.isnan(OP)]=0.0

    # 積分の実行
    IFT2=np.fft.ifft2(FA*OP)
    return np.real(IFT2[0:n, 0:n])    
    

    
Moire = np.load('Moire.npy')

ABS, DPx, DPy, SPx, SPy = FFTDemod(Moire)
IntPhase = Integration(DPx, DPy)


plt.imshow(ABS, cmap='bone')
plt.savefig('ABSImg.png')    
plt.clf()

plt.imshow(DPx, cmap='bone')
plt.savefig('DPxImg.png')    
plt.clf()

plt.imshow(DPy, cmap='bone')
plt.savefig('DPyImg.png')    
plt.clf()

plt.imshow(SPx, cmap='bone')
plt.savefig('SPxImg.png')    
plt.clf()

plt.imshow(SPy, cmap='bone')
plt.savefig('SPyImg.png')    
plt.clf()

plt.imshow(IntPhase, cmap='bone')
plt.savefig('IntPhaseImg.png')    
plt.clf()
