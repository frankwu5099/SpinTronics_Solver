
import tensorflow as tf

def tf_diff_x(X):
    return tf.manip.roll(X,-1,axis = 0 ) - tf.manip.roll(X,1,axis = 0)

def tf_diff_y(X):
    return tf.manip.roll(X,-1,axis = 1 ) - tf.manip.roll(X,1,axis = 1)

def tf_diff_z(X):
    #A =  tf.manip.roll(X,1,axis = 2 ) - tf.manip.roll(X,-1,axis = 2)
    #A [:,:,0] = 0
    return X[:,:,2:] -X[:,:,:-2]

def tf_diff_z2(X,btop):
    return tf.concat([-1.5*X[:,:,0:1] +2*X[:,:,1:2] -0.5*X[:,:,2:3] ,X[:,:,2:] -X[:,:,:-2], 1.5*X[:,:,-1:] -2*X[:,:,-2:-1] +0.5*X[:,:,-3:-2]],axis = 2)

def J_I(D ,Gr, Gi, a , Sx , Sy, Sz, mx, my, mz, moving): #Jxz Jyz Jzz
    r"""
    Implement of the equation: 
    $J_s^{interface} = - G_r \bm m \times (\bm m \times \bm S)- G_i (\bm m \times \bm S)$
    """
    mx_tmp = tf.roll(mx,moving,axis=1)
    my_tmp = tf.roll(my,moving,axis=1)
    mz_tmp = tf.roll(mz,moving,axis=1)
    cross_x = Sy[:,:,:1] * mz_tmp - Sz[:,:,:1] * my_tmp
    cross_y = Sz[:,:,:1] * mx_tmp - Sx[:,:,:1] * mz_tmp
    cross_z = Sx[:,:,:1] * my_tmp - Sy[:,:,:1] * mx_tmp
    return -(Gi*cross_x + Gr*(my_tmp*cross_z - mz_tmp*cross_y)),\
        -(Gi*cross_y + Gr*(mz_tmp*cross_x - mx_tmp*cross_z)),\
        -(Gi*cross_z + Gr*(mx_tmp*cross_y - my_tmp*cross_x))

def dSdt (D ,lsf2, Gr, Gi, a, E_reduced, mx, my, mz, btop, moving, t, Sx , Sy, Sz):
    r"""
    Implement of the equation: 
    $\partial \bm S / \partial t = -\nabla \cdot \bmJ_{\bm S} - D (\bm S/\lambda_f)$
    """
    Jxx, Jxy, Jxz, Jyx, Jyy, Jyz, Jzx, Jzy, Jzz = Jbulk(D ,lsf2, a , Sx , Sy, Sz, E_reduced)
    Jxz0, Jyz0, Jzz0 = J_I(D ,Gr, Gi, a , Sx , Sy, Sz, mx, my, mz, moving)
    Jxz = tf.concat([Jxz0,Jxz,btop],axis = 2)
    Jyz = tf.concat([Jyz0,Jyz,btop],axis = 2)
    Jzz = tf.concat([Jzz0,Jzz,btop],axis = 2)
    return  -D * (Sx/lsf2) -tf_diff_x(Jxx)/a -tf_diff_y(Jxy)/a -tf_diff_z2(Jxz,btop)/a,\
            -D * (Sy/lsf2) -tf_diff_x(Jyx)/a -tf_diff_y(Jyy)/a -tf_diff_z2(Jyz,btop)/a,\
            -D * (Sz/lsf2) -tf_diff_x(Jzx)/a -tf_diff_y(Jzy)/a -tf_diff_z2(Jzz,btop)/a


def Js (D ,lsf2, Gr, Gi, a, E_reduced, mx, my, mz, btop, t, Sx , Sy, Sz, moving):
    r"""
    Calculate $J_s^{interface}$ and $J_s^{bulk}$ and sum them up.
    Calculate the Hall current contributed by spin current.
    """
    Jxx, Jxy, Jxz, Jyx, Jyy, Jyz, Jzx, Jzy, Jzz = Jbulk(D ,lsf2, a , Sx , Sy, Sz, E_reduced)
    Jxz0, Jyz0, Jzz0 = J_I(D ,Gr, Gi, a , Sx, Sy, Sz, mx, my, mz, moving)
    Jxz = tf.concat([Jxz0,Jxz,btop],axis = 2)
    Jyz = tf.concat([Jyz0,Jyz,btop],axis = 2)
    Jzz = tf.concat([Jzz0,Jzz,btop],axis = 2)
    return tf.reduce_sum(Jyz - Jzy) , tf.reduce_sum(Jzx - Jxz)

def dSdt_I (D , Gr, Gi, a , Sx , Sy, Sz, mx, my, mz):
    cross_x = Sy[:,:,:1] * mz - Sz[:,:,:1] * my
    cross_y = Sz[:,:,:1] * mx - Sx[:,:,:1] * mz
    cross_z = Sx[:,:,:1] * my - Sy[:,:,:1] * mx
    return -(Gi * cross_x + Gr * (my*cross_z - mz*cross_y)),\
        -(Gi * cross_y + Gr * (mz*cross_x - mx*cross_z)),\
        -(Gi * cross_z + Gr * (mx*cross_y - my*cross_x))


def Jbulk (D ,lsf2, a , Sx , Sy, Sz, E_reduced):
    r"""
    Implement of the equation: 
    $J_s^{bulk} = - D \nabla S + spin hall term$
    """
    return -D * tf_diff_x(Sx)/a,\
        -D * tf_diff_y(Sx)/a,\
        -D * tf_diff_z(Sx)/a,\
        -D * tf_diff_x(Sy)/a,\
        -D * tf_diff_y(Sy)/a,\
        -D * tf_diff_z(Sy)/a + E_reduced,\
        -D * tf_diff_x(Sz)/a,\
        -D * tf_diff_y(Sz)/a - E_reduced,\
        -D * tf_diff_z(Sz)/a


