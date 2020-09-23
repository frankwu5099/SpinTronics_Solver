
import tensorflow as tf
def rk4_step(time_derivative, t, x, y, z, h):
    with tf.variable_scope('k1'):
        kx1,ky1,kz1 = time_derivative(t, x, y, z)
        print (kx1.shape)
    with tf.variable_scope('k2'):
        with tf.variable_scope('t1'):
            t1 = t + h/2;
        with tf.variable_scope('x1'):
            x1 = x + h*kx1/2;
            y1 = y + h*ky1/2;
            z1 = z + h*kz1/2;
        kx2,ky2,kz2 = time_derivative(t1, x1, y1, z1)
    with tf.variable_scope('k3'):
        with tf.variable_scope('t2'):
            t2 = t + h/2;
        with tf.variable_scope('x2'):
            x2 = x + h*kx2/2;
            y2 = y + h*ky2/2;
            z2 = z + h*kz2/2;
        kx3,ky3,kz3 = time_derivative(t2, x2, y2, z2)
    with tf.variable_scope('k4'):
        with tf.variable_scope('t3'):
            t3 = t + h;
        with tf.variable_scope('x3'):
            x3 = x + h*kx3;
            y3 = y + h*ky3;
            z3 = z + h*kz3;
        kx4,ky4,kz4 = time_derivative(t3, x3, y3, z3)
    
    with tf.variable_scope('new_state'):
        return h/6 * (kx1 + kx2*2 + kx3*2 + kx4) , h/6 * (ky1 + ky2*2 + ky3*2 + ky4) , h/6 * (kz1 + kz2*2 + kz3*2 + kz4)

def update_state(x,y,z,t,dx,dy,dz,tf_dt):
    with tf.variable_scope('update_state'):
        return tf.group(tf.assign_add(x, dx),tf.assign_add(y, dy),tf.assign_add(z, dz),
            tf.assign_add(t, tf_dt),
            name='update_state')