import numpy as np
import curses

class Framebuffer:
    MAX_DEPTH = 128
    SHADES = "@#&%*+-'`. "
    SHADES_COUNT = len(SHADES)-1

    def __init__(self, scr, id):
        self.scr = scr
        self.id = id
        self.height, self.width = scr.getmaxyx()
        self.half_shape = np.array([self.width, self.height]) / 2
        self.depth_buffer = np.zeros((self.width, self.height), dtype=np.uint8)
        #self.color_buffer = np.zeros((self.width, self.height), dtype=np.uint8)
        self.symbol_buffer = np.full((self.width, self.height), " ", dtype="U1")

    def print(self):
        # TODO ITS BETTER TO PREPEARE WHOLE STRING AND PRINT IT
        try:
            np.apply_along_axis(lambda p: self.scr.insch(p[1], p[0], "@"),
                                axis=1,
                                arr=np.argwhere(self.symbol_buffer != " "))
        except ValueError:
            # Nothing is visible
            pass
        return True

    def clear(self):
        self.depth_buffer.fill(0)
        #self.color_buffer.fill(0)
        #self.symbol_buffer.fill(" ")
