import concurrent.futures
import numpy as np

from transformation import Transformation
from framebuffer import Framebuffer
        
class Rasterizer:
    
    def __init__(self, scr):
        self.scr = scr
        self.fbs = [Framebuffer(scr, id=0), Framebuffer(scr, id=1)]
        self.current_buffer = 0
        self.obj = None
        self.print_buffer = self.fbs[0]
        self.render_buffer = self.fbs[1]
        self.exec = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.swap_buffers()
    
    def add_model(self, obj):
        self.obj = obj

    def swap_buffers(self,):
        try:
            if self.render_future.result() and self.print_future.result():
                self.current_buffer = not self.current_buffer   
                self.print_buffer = self.fbs[self.current_buffer]
                self.render_buffer = self.fbs[not self.current_buffer]    
        except AttributeError:
            # First render
            pass  
                
    def print_frame(self):
        self.print_future = self.exec.submit(self.print_buffer.print)
        self.render_future = self.exec.submit(self.obj.render, self.render_buffer)
                    
   
        
