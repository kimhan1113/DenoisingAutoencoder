import numpy as np
import cv2
from scipy import ndimage

class CalcStat:

    def __init__(self):
        self.OK = None
        return 0

    def RMS(self, src, ref):
        res = np.sqrt( np.sum(np.square(np.subtract(src, ref))) / len(src))
        return res

    def Off(self, src, ref):

        src_array2D = np.array(src)
        ref_array2D = np.array(ref)

        src_array1D = src_array2D.reshape(src_array2D.shape[0] * src_array2D.shape[1])
        ref_array1D = ref_array2D.reshape(ref_array2D.shape[0] * ref_array2D.shape[1])

        src_list = src_array1D.tolist()
        ref_list = ref_array1D.tolist()

        src_list.sort(reverse=True)
        ref_list.sort(reverse=True)

        for i in range(0,int(len(src_list) * 0.03)):
            src_list.pop(i)
            ref_list.pop(i)

        src_list_8 = list()
        ref_list_8 = list()

        for i in range(0, int(len(src_list) * 0.08)):
            src_list_8.append(src_list[i])
            ref_list_8.append(ref_list[i])

        src_mean = np.mean(np.array(src_list_8))
        # src_stv = np.stv(np.array(src_list_8))

        ref_mean = np.mean(np.array(ref_list_8))
        # ref_stv = np.stv(np.array(ref_list_8))

        res = abs(src_mean - ref_mean)

        # res = np.sum(abs(np.subtract(src, ref))) / len(src)

        return res

    def SAG(self, src):
        kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ky = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

        Gx = ndimage.convolve(src, kx, mode='constant', cval=0.0)
        Gy = ndimage.convolve(src, ky, mode='constant', cval=0.0)

        res = np.sqrt(np.sum(np.add(np.square(Gx), np.square(Gy))) / len(src))

        return res




