import numpy as np

def get_comp_data(data,point,rad_size):
    return np.average(data[:,point[0]-rad_size:point[0]+rad_size,point[1]-rad_size:point[1]+rad_size],axis=(1,2))

def get_circ_centers(file_path,flipy = True):
    with open(file_path,'r') as f:
        lines = f.readlines()
        points = np.zeros((len(lines)-1,2))
#         diameters = np.zeros((len(lines)-1,1))
        for i,line in enumerate(lines[1:]):
            l = line.split(',')
            points[i] = np.array([float(l[1]),float(l[0])])
#             diameters[i] = np.array([float(l[2])])
            if (flipy):
                points[i,0] = -points[i,0]
    print ('extracted points: '+ str(len(points)))
    return points#,diameters

def affine_unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def affine_angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::"""
    v1_u = affine_unit_vector(v1)
    v2_u = affine_unit_vector(v2)
    r =  np.math.atan2(np.linalg.det([v1_u,v2_u]),np.dot(v1_u,v2_u))
    cosr = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    sinr = np.linalg.det([v1_u,v2_u])
    return r, cosr ,sinr

def affine_get_transform(src,tgt):
    tt = -np.mean(src,axis = 0)
    t  = np.mean(tgt,axis = 0)
    s = np.linalg.norm(tgt[1]-tgt[0])/np.linalg.norm(src[1]-src[0])
    r, cosr ,sinr = affine_angle_between(src[0]-src[1],tgt[0]-tgt[1])
    print('-------transform -------------------')
    print ('scale: %.6f, rotation: %.4f[deg]'%(s,180*r/np.pi))
    print ('src translations: ' + format(tt))
    print ('tgt translations: ' + format(t))
    print('------------------------------------')
    A2 = np.matrix([[s*cosr,-s*sinr,s*(tt[0]*cosr-tt[1]*sinr)+t[0]],
                    [s*sinr, s*cosr,s*(tt[0]*sinr+tt[1]*cosr)+t[1]],
                    [0     ,0      ,1                             ]])
    return lambda x: np.asarray((A2*np.vstack((np.matrix(x).reshape(2,1),1)))[0:2,:]).flatten()
