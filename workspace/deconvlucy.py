import imageio
import numba as nb
import numpy as np
from numba import njit
from scipy.signal import convolve
from skimage.transform import rescale


def deconvlucy_iter(im, im_deconv, psf, psf_mirror):
    relative_blur = im / convolve(im_deconv, psf, mode="same")
    im_deconv *= convolve(relative_blur, psf_mirror, mode="same")
    return im_deconv


def deconvlucy(image, psf, iterations, clip=False):
    im_deconv = np.full(image.shape, 0.5)
    psf_mirror = psf[::1, ::-1, ::-1]

    for i in range(iterations):
        print(f"iter {i+1}")
        im_deconv = deconvlucy_iter(image, im_deconv, psf, psf_mirror)

    if clip:
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < -1] = -1

    return im_deconv.astype(np.float32)


def main(image_path, psf_path):
    image = imageio.volread(image_path)  # z 0.6u
    psf = imageio.volread(psf_path)  # z 0.2u

    image, psf = image.astype(np.float32), psf.astype(np.float32)

    # crop image (too large)
    image = image[:, 768 : 768 + 512, 768 : 768 + 512]

    imageio.volwrite("input.tif", image)
    raise RuntimeError("DEBUG")

    # rescale psf
    psf = rescale(psf, (1 / 3, 1, 1), anti_aliasing=False)
    # normalize
    psf = (psf - psf.max()) / (psf.max() - psf.min())
    psf /= psf.sum()
    print(f"psf range [{psf.min(), psf.max()}]")

    print(f"image shape {image.shape}, psf shape {psf.shape}")

    try:
        deconv = deconvlucy(image, psf, 10)
    except Exception:
        raise
    else:
        print("saving...")
        imageio.volwrite("result.tif", deconv)


if __name__ == "__main__":
    main(
        "C:/Users/Andy/Desktop/background_removal/flybrain_Iter_ch0_stack0000_640nm_0000000msec_0015869850msecAbs.tif",
        "C:/Users/Andy/Desktop/background_removal/psf/NA1p05_zp6um_cropped.tif",
    )
