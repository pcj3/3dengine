import numpy as np
from transformation import Transformation

class Camera:
    
    def __init__(self):
        self.rotation = np.array([0, 0, 0]).reshape(3, 1)
        self.position = np.array([0, 10, 0]).reshape(3, 1)
        self.light = np.array([0, 0, 10]).reshape(3, 1)
        self.view = Transformation()
        self.proj = Transformation()
        
    @property    
    def proj_view(self):
        return self.view.arr @ self.proj.arr
        
    def calculate_view(self):
        self.view.reset()
        self.view.rotate(x=1, y=0, z=0, angle=self.rotation[0])
        self.view.rotate(x=0, y=1, z=0, angle=self.rotation[1])
        self.view.rotate(x=0, y=0, z=1, angle=self.rotation[2])
        self.view.translate(0, 0, 0)
        
    def create_proj(self, fov, aspect_ratio, near_plane, far_plane):
        self.proj.reset()
        tan_half_fov = np.tan(fov / 2)
        self.proj.arr[0 ,0] = 1 / (aspect_ratio * tan_half_fov)
        self.proj.arr[1, 1] = 1 / tan_half_fov
        self.proj.arr[2, 2] = -(near_plane + far_plane) / (far_plane - near_plane)
        self.proj.arr[2, 3] = -1
        self.proj.arr[3, 2] = -(2 * near_plane * far_plane) / (far_plane - near_plane)
    
