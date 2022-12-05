import numpy as np

class Transformation:
    
    def __init__(self):
        self.arr = np.eye(4)
        
    def scale(self, x, y, z):
        self.arr[:3, :] = self.arr[:3, :] * np.array([x, y, z]).reshape((3, 1))
        
    def rotate(self, x, y, z, angle):
        axis = np.array([x, y, z]).reshape((3,1))
        axis = self._normalize(axis)
        axis_cross_product_mat = np.cross(axis.squeeze(), np.identity(axis.shape[0]) * -1)                    
        rot_mat = np.cos(angle) * np.eye(3) \
                    + np.sin(angle) * axis_cross_product_mat \
                    + (1 - np.cos(angle)) * np.outer(axis, axis)
        self.arr[:3,:] = rot_mat.T @ self.arr[:3,:]
        
    def translate(self, x, y, z):
        vect = np.array([x, y, z]).reshape((3,1))
        self.arr[3,:] += (self.arr.T[:,:3] @ vect).squeeze()
                        
    def reset(self):
        self.arr = np.eye(4)
    
    @staticmethod    
    def _normalize(vect):
        if (mag := np.linalg.norm(vect)) != 0:
            return vect / mag
        return vect
        