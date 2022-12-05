import numpy as np

from transformation import Transformation

class Model:
    
    def __init__(self, camera, file_path):            
        self.trans = Transformation()
        self.cam = camera
        self.pos_indicies, self.model_pos, self.norm_indicies, self.norms  = self._load_file(file_path)
        self.proj_pos = None
        self.scr_pos = None
        
    def _calc_normals(self, ):
        print(self.indicies)
        
    def _load_file(self, file_path): 
        with open(file_path, "r") as file:
            lines = file.readlines()
        first_char_lines = [line[:2] for line in lines]
        
        first_v_idx = first_char_lines.index("v ")
        last_v_idx = len(first_char_lines) - first_char_lines[::-1].index("v ")
        positions = np.genfromtxt(lines[first_v_idx:last_v_idx], usecols=(1, 2, 3))
        positions = np.hstack([positions, np.ones((positions.shape[0], 1))])
        
        try:
            first_vt_idx = first_char_lines.index("vt")
            last_vt_idx = len(first_char_lines) - first_char_lines[::-1].index("vt")
            uvs = np.genfromtxt(lines[first_vt_idx:last_vt_idx], usecols=(1, 2))
        except ValueError:
            # No uvs data TODO
            uvs = None
            
        try:
            first_vn_idx = first_char_lines.index("vn")
            last_vn_idx = len(first_char_lines) - first_char_lines[::-1].index("vn")
            normals = np.genfromtxt(lines[first_vn_idx:last_vn_idx], usecols=(1, 2, 3))
            normals =  np.hstack([normals, np.zeros((normals.shape[0], 1))])
        except ValueError:
            # No normals data TODO
            normals = None      
            norm_indicies = None
            
        first_f_idx = first_char_lines.index("f ")
        last_f_idx = len(first_char_lines) - first_char_lines[::-1].index("f ")
        
        if uvs is not None and normals is not None:
            f_lines = [item[2:-1].replace("/", " ") for item in lines[first_f_idx:last_f_idx]]
            all_indicies = np.genfromtxt(f_lines, dtype=np.int32).reshape(len(f_lines), 3, 3) - 1
            pos_indicies = all_indicies[:,:,0]
            norm_indicies = all_indicies[:,:,2]
        else:
            f_lines = lines[first_f_idx:last_f_idx]
            pos_indicies = np.genfromtxt(f_lines, usecols=(1, 2, 3),
                            dtype=np.int32) - 1
        return pos_indicies, positions, norm_indicies, normals

                
    def _points_inside_face(self, barycentric_xx, barycentric_yy, barycentric_zz):
        return np.logical_and(np.logical_xor(~(barycentric_xx<-0.001), barycentric_yy<-0.001), 
                                np.logical_xor(~(barycentric_yy<-0.001), barycentric_zz<-0.001))        
    
    def _calc_barycentric_coords(self, xx, yy, vertices):
        v1, v2, v3 = vertices
        vect32 = v2 - v3
        vect31 = v1 - v3
        vectx3 = xx - v3[0]
        vecty3 = yy - v3[1]
        denominator = vect32[1] * vect31[0] - vect32[0] * vect31[1]
        barycentric_x = (vect32[1] * vectx3 - vect32[0] * vecty3) / denominator
        barycentric_y = (-vect31[1] * vectx3 + vect31[0] * vecty3) / denominator
        barycentric_z = 1 - barycentric_x - barycentric_y
        return barycentric_x, barycentric_y, barycentric_z
        

    def _calc_proj_barycentric_coords(self, xx, yy, zz, vertices):
        v1, v2, v3 = vertices
        denominator = (xx / v1[3]) + (yy / v2[3]) + (zz / v3[3])
        scr_bc_xx = (xx / v1[2]) / denominator
        scr_bc_yy = (yy / v2[2]) / denominator
        scr_bc_zz = (zz / v3[2]) / denominator
        return scr_bc_xx, scr_bc_yy, scr_bc_zz
        

    def render(self, render_buffer):
        render_buffer.clear()
        model_view_projection =  self.trans.arr @ self.cam.proj_view 
        self.proj_pos = (model_view_projection.T @ self.model_pos.T).T
        self.proj_pos[:,:2] /= self.proj_pos[:,2][:, np.newaxis]
        self.world_pos = (self.trans.arr.T @ self.model_pos.T).T 
        self.trans_norms = (self.trans.arr.T @ self.norms.T).T        
        self.light = self.cam.light.squeeze() - self.world_pos[:,:3]
        self.scr_pos = self.proj_pos[:,:2] * [1, -1] * render_buffer.half_shape + render_buffer.half_shape          
        for i in range(self.pos_indicies.shape[0]):
            self._render_face(self.pos_indicies[i], self.norm_indicies[i], render_buffer)        
        # Return True for threding check    
        return True
           
    def _render_face(self, pos_idx, norm_idx, render_buffer):
        #print(idx)
        minx = np.clip(self.scr_pos[pos_idx, 0].min(), 0, None).astype(np.int16)
        miny = np.clip(self.scr_pos[pos_idx, 1].min(), 0, None).astype(np.int16)
        maxx = np.clip(self.scr_pos[pos_idx, 0].max()+1, None, render_buffer.width).astype(np.int16)
        maxy = np.clip(self.scr_pos[pos_idx, 1].max()+1, None, render_buffer.height).astype(np.int16)
        xx, yy = np.meshgrid(np.arange(minx, maxx), 
                            np.arange(miny, maxy),
                            sparse=True,
                            indexing="ij")
                
        bc_v1, bc_v2, bc_v3 = self._calc_barycentric_coords(xx, yy, self.scr_pos[pos_idx])
        proj_bc_v1, proj_bc_v2, proj_bc_v3 = self._calc_proj_barycentric_coords(bc_v1, bc_v2, bc_v3, self.proj_pos[pos_idx])
            
        depths = (self.proj_pos[pos_idx[0], 2] / self.proj_pos[pos_idx[0], 3]) * bc_v1 \
                + (self.proj_pos[pos_idx[1], 2] / self.proj_pos[pos_idx[1], 3]) * bc_v2 \
                + (self.proj_pos[pos_idx[2], 2] / self.proj_pos[pos_idx[2], 3]) * bc_v3
        depths *= render_buffer.MAX_DEPTH

        pts_inside = self._points_inside_face(bc_v1, bc_v2, bc_v3)
        render_area = render_buffer.depth_buffer[minx:maxx, miny:maxy]
        mask = np.logical_and(pts_inside, depths >= render_area)
    
        face_norm = self.trans_norms[norm_idx[0]]
        face_norm_xx = face_norm[0] * proj_bc_v1 + face_norm[0] * proj_bc_v2 + face_norm[0] * proj_bc_v3
        face_norm_yy = face_norm[1] * proj_bc_v1 + face_norm[1] * proj_bc_v2 + face_norm[1] * proj_bc_v3
        face_norm_zz = face_norm[2] * proj_bc_v1 + face_norm[2] * proj_bc_v2 + face_norm[2] * proj_bc_v3
        face_norm_mag = np.sqrt(face_norm_xx ** 2 + face_norm_yy **2 + face_norm_zz ** 2)
        face_norm_xx /= face_norm_mag
        face_norm_yy /= face_norm_mag
        face_norm_zz /= face_norm_mag
        
        light_xx = self.light[pos_idx[0], 0] * proj_bc_v1 + self.light[pos_idx[1], 0] * proj_bc_v2 + self.light[pos_idx[2], 0] * proj_bc_v3
        light_yy = self.light[pos_idx[0], 1] * proj_bc_v1 + self.light[pos_idx[1], 1] * proj_bc_v2 + self.light[pos_idx[2], 1] * proj_bc_v3
        light_zz = self.light[pos_idx[0], 2] * proj_bc_v1 + self.light[pos_idx[1], 2] * proj_bc_v2 + self.light[pos_idx[2], 2] * proj_bc_v3
        light_mag = np.sqrt(light_xx ** 2 + light_yy **2 + light_zz ** 2)
        light_xx /= light_mag
        light_yy /= light_mag
        light_zz /= light_mag

        brightness = face_norm_xx * light_xx + face_norm_yy * light_yy + face_norm_zz * light_zz     
        brightness = 1 - np.clip(brightness, None, .19)
        brightness *= render_buffer.SHADES_COUNT
        brightness = brightness.astype(np.int8)
        print(brightness)
        shades = np.array([render_buffer.SHADES[i] for i in brightness.flatten()], dtype="U1").reshape(brightness.shape)
        print(shades)
        try:
            print(":)")
            render_buffer.symbol_buffer[minx:maxx, miny:maxy][mask] = "#"
            print(":D")
        except ValueError:
            # Not in camera planes scope
            pass
            
if __name__ == "__main__":
    model = Model(1, r"res/cube.obj")        