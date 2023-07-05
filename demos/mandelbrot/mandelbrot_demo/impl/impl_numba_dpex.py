import dpnp as np
import numba
from mandelbrot_demo.impl.settings import MAX_ITER
from numba_dpex import dpjit


@dpjit
def mandelbrot(c1, c2, c3, w, h, zoom, offsetx, offsety, values):
    for x in numba.prange(w):
        for y in range(h):
            xx = (x - offsetx) * zoom
            yy = (y - offsety) * zoom
            cReal = xx
            cImage = yy
            zReal = 0
            zImage = 0
            mand = -1
            for i in range(MAX_ITER):
                zReal = zReal * zReal - zImage * zImage + cReal
                zImage = zReal * zImage * 2 + cImage
                if (zReal * zReal + zImage * zImage) > 4.0:
                    mand = i
                    break
            if mand == -1:
                mand = MAX_ITER
            intensity = mand / MAX_ITER
            for c in range(3):
                if intensity < 0.5:
                    color = c3[c] * intensity + c2[c] * (1.0 - intensity)
                else:
                    color = c1[c] * intensity + c2[c] * (1.0 - intensity)
                color = int(color * 255.0)
                values[x, y, c] = color
    return values


def init_values(w, h):
    return np.full((w, h, 3), 0, dtype=np.uint8)


def asnumpy(values):
    return np.asnumpy(values)
