

import numpy as np
import tensorflow as tf
from functools import partial
from EOM import *
from RK import *
# Just disables the warning, doesn't enable AVX/FMA
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
np.set_printoptions(threshold=sys.maxsize)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def topo_charge(sx, sy, sz):
    """
    find out the topological charge (skyrmion number) for configuration (sx, sy, sz)
    """
    rf = np.sqrt(sx*sx + sy*sy + sz*sz)
    sx_p = sx/rf
    sy_p = sy/rf
    sz_p = sz/rf
    dxsx = np.roll(sx_p,-1,axis = 0)# - sx_p
    dxsy = np.roll(sy_p,-1,axis = 0)# - sy_p
    dxsz = np.roll(sz_p,-1,axis = 0)# - sz_p
    dysx = np.roll(sx_p,-1,axis = 1)# - sx_p
    dysy = np.roll(sy_p,-1,axis = 1)# - sy_p
    dysz = np.roll(sz_p,-1,axis = 1)# - sz_p
    vol = sx_p * dxsy * dysz - sx_p * dysy * dxsz 
    vol += sy_p * dxsz * dysx - sy_p * dysz * dxsx 
    vol += sz_p * dxsx * dysy - sz_p * dysx * dxsy 
    div = 1.+ sx_p*dxsx+ sy_p*dxsy+ sz_p*dxsz+ sx_p*dysx+ sy_p*dysy+ sz_p*dysz+ dxsx*dysx+ dxsy*dysy+ dxsz*dysz
    C = np.arctan(vol/div)
    sx_p = np.roll(np.roll(sx_p, -1, axis = 0), -1, axis = 1)
    sy_p = np.roll(np.roll(sy_p, -1, axis = 0), -1, axis = 1)
    sz_p = np.roll(np.roll(sz_p, -1, axis = 0), -1, axis = 1)
    vol = sx_p * dxsy * dysz - sx_p * dysy * dxsz 
    vol += sy_p * dxsz * dysx - sy_p * dysz * dxsx 
    vol += sz_p * dxsx * dysy - sz_p * dysx * dxsy 
    div = 1.+ sx_p*dxsx+ sy_p*dxsy+ sz_p*dxsz+ sx_p*dysx+ sy_p*dysy+ sz_p*dysz+ dxsx*dysx+ dxsy*dysy+ dxsz*dysz
    C -= np.arctan(vol/div)
    return C.sum()
#use the unit: nm 10^-7s

a = 0.2 # unit cell size
E = 0.01 # it is a combination factor of spin hall angle, muB_e and electric field (should be fixed) 
muB_e = 0.00562 #nm^2/(10^-16s)

lsf = 1.2 # \lambda_sf -- spin diffusion length
lsf2 = lsf*lsf 
D = 1 # diffusion coefficient 
# th_SHA 
# spin mixing conductance
Gr = 1.0 
Gi = 0.1 

tf.reset_default_graph()
sess = tf.InteractiveSession()



# setup time and time_step
dt =0.003
tf_dt = tf.constant(dt, dtype=tf.float32, name='tf_dt')
t = tf.Variable(0, dtype=tf.float32)


# setting the system size for the normal metal part
Lx, Ly, Lz = 80, 80, 40

# --------- initializing skyrmion configuration -----------
mx00  = np.zeros((Lx,Ly,1))
my00  = np.zeros((Lx,Ly,1)) 
mz00  = np.ones((Lx,Ly,1))

Lsk = 20
xaxis = np.linspace(-Lx/2,Lx/2,Lx)
yaxis = np.linspace(-Ly/2,Ly/2,Ly)

zaxis_I = [0.0]
xv, yv, zv = np.meshgrid(xaxis ,yaxis,zaxis_I, indexing = 'ij')

# following code are for future 
#zaxis = np.linspace( 0 ,Lz,Lz)
#xv3d, yv3d, zv3d = np.meshgrid(xaxis,yaxis,zaxis, indexing = 'ij')

theta = 2.0*np.arctan(np.sinh(4)/np.sinh(4*np.sqrt(xv*xv +yv*yv) /Lsk))
phi = np.arctan2(yv,xv)
mx00 = np.sin(theta) * np.cos(phi)
my00 = np.sin(theta) * np.sin(phi)
mz00 = np.cos(theta)
mx0 = mx00 * 1.0
my0 = my00 * 1.0
mz0 = mz00 * 1.0

plt.quiver(xv[::,::,0], yv[::,::,0], mx0[::,::,0], my0[::,::,0], mz0[::,::,0], alpha=.5,cmap = cm.coolwarm, pivot = 'middle')
plt.colorbar()
plt.show()
plt.quiver(xv[:,:-1,0], yv[:,:-1,0], mx0[:,:-1,0] - mx0[:,1:,0], my0[:,:-1,0] - my0[:,1:,0], mz0[:,:-1,0] - mz0[:,1:,0],
           alpha=.5,cmap = cm.coolwarm, pivot = 'middle')
plt.colorbar()
plt.show()
print("average magnetization: ", mz0.mean())
mx = tf.constant(mx0, dtype=tf.float32, name='mag_x')
my = tf.constant(my0, dtype=tf.float32, name='mag_y')
mz = tf.constant(mz0, dtype=tf.float32, name='mag_z')
# ------- end initializing skyrmion configuration ---------

# initializing the spin accumulation
btop  = np.zeros((Lx,Ly,1)) #vacuum boundary condition on one side
# ***** spin accumulation *****
Sx0  = np.zeros((Lx,Ly,Lz))
Sy0  = 0.0*np.ones((Lx,Ly,Lz)) 
Sz0  = np.zeros((Lx,Ly,Lz))
# ***** ***************** *****
Sx = tf.Variable(Sx0, dtype=tf.float32)
Sy = tf.Variable(Sy0, dtype=tf.float32)
Sz = tf.Variable(Sz0, dtype=tf.float32)

# moving skrymion 
moving0 = 0
moving = tf.Variable(moving0, dtype=tf.int32)

dSdt_ = partial(dSdt, D, lsf2, Gr, Gi, a, E, mx, my, mz, btop, moving) # setup the time derivative for spin accumulation
dSx, dSy, dSz = rk4_step(dSdt_, t, Sx, Sy, Sz, tf_dt) # estimate dS(t) by RK4
update = update_state(Sx, Sy, Sz, t, dSx, dSy, dSz, tf_dt) # update S(t) -> S(t+dt) = S(t) + dS(t)
Jcout = Js (D ,lsf2, Gr, Gi, a, E, mx, my, mz, btop, t, Sx , Sy, Sz, moving) # get spin current

# tf ininitialize the configuration
sess.run(tf.global_variables_initializer())

for i in range(20000):
    #movement of texture
    #assign = moving.assign(4*i)
    #sess.run(assign)

    sess.run([update]) #update a time step

    if i%1000 == 999:
        sxout, syout, szout = sess.run([Sx,Sy,Sz])
        for i_layer in range(40):
                syout[:,:,i_layer] = syout[:,:,i_layer]
                szout[:,:,i_layer] = szout[:,:,i_layer]
                sxout[:,:,i_layer] = sxout[:,:,i_layer]
                di = 2
                r = np.sqrt(sxout[::di,::di,i_layer]*sxout[::di,::di,i_layer] +
                            syout[::di,::di,i_layer]*syout[::di,::di,i_layer] +
                            szout[::di,::di,i_layer]*szout[::di,::di,i_layer])
                plt.quiver(xv[::di,::di,0], yv[::di,::di,0],
                           sxout[::di,::di,i_layer], syout[::di,::di,i_layer], szout[::di,::di,i_layer],
                           alpha=.5,cmap = cm.coolwarm, pivot = 'middle')
                plt.colorbar()
                plt.show()
                Nsk = topo_charge(sxout[:,:,i_layer], syout[:,:,i_layer], szout[:,:,i_layer])
                print (Nsk, Nsk /Lx /Ly)
        J_long, J_trans = sess.run([Jcout])[0]
        print("Jtrans:", J_trans/Lx/Ly/Lz, "Jlong:", J_long/Lx/Ly/Lz)
        #print(syout)
#plt.plot(syout.flatten())
