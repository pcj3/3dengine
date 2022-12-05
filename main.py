import curses 
import numpy as np

from rasterizer import Rasterizer
from camera import Camera
from model import Model

def main(stdscr, obj_file_path):
    curses.curs_set(0)
    height, width = stdscr.getmaxyx()
    rizer = Rasterizer(stdscr)
    cam = Camera()
    cam.create_proj(fov=1.2, 
                    aspect_ratio=(width / 2) / height,
                    near_plane=0.2,
                    far_plane=400)
    cam.calculate_view()
    model = Model(cam, obj_file_path)                    
    rizer.add_model(model)
    i, angle = 0, 0  
    while True:
        i += 1
        #For debugging
        pressed_key = stdscr.getch()
        if pressed_key == ord("q"):
            break      
        if pressed_key == ord("d"):
            cam.position = np.array([0, 10, 0]).reshape(3, 1)
            
        stdscr.erase() 
        model.trans.reset()
        
        model.trans.translate(0, 0, -2) # This is obj depended
        #model.trans.rotate(0, 1, 0, angle) 
        #model.trans.scale(1, 1, 1) # This is obj depended
        
        angle += .005
        stdscr.addstr(height-1, 0, str(i))
        rizer.print_frame()
        rizer.swap_buffers()
        stdscr.refresh()
        #stdscr.erase() # Comment this for debugging
        
        

if __name__ == "__main__":
    obj_file_path = r".\res\cube.obj"
    curses.wrapper(main, obj_file_path)