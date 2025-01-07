from v4 import vx
from numpy import *
## vx input buffer
class vxIbuf:
    def __init__(self, *args):
        if 2 == len(args) :
            self.n = args[1]
            self.vx = vx.Vx( args[0])
            self.bn = 0;
            tshape = self.vx.i.shape
            self.mx = tshape[0]
            self.i = empty([self.n, tshape[1],tshape[2]], dtype=self.vx.i.dtype)
    def read (self):
        ##check if any data?
        if self.mx - self.bn >= self.n :
            #update data
            for i in range(self.n):
                self.i[i] = self.vx.i[self.bn + i]
            self.bn += 1
            return True
        else :
            return False
                            
## vx outbuffer    
class vxObuf:
    def __init__(self, *args):
        if 1 == len(args) :
            self.wfile = args[0]
            self.n = 0
    def add(self, imarray):
        if ( self.n == 0 ):
            ##self.i = imarray[newaxis]
            self.i = expand_dims(imarray, axis=0)
        else:
            ##tmp = expand_dims(imarray, axis=0)
            self.i = concatenate((self.i, expand_dims(imarray, axis=0)) , axis=0)
        self.n += 1
    def close(self):
        self.sh = [0,self.i.shape[2],0,self.i.shape[1],0,self.i.shape[0]]
        self.vx = vx.Vx(self.i.dtype, self.sh)
        self.vx.setim(self.i)
        self.vx.write(self.wfile)
