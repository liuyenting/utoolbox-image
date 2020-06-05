import imageio
import numpy as np
from scipy.signal import convolve
from skimage.transform import rescale


def deconvlucy(image, psf, iterations, clip=True):
    im_deconv = np.full(image.shape, 0.5)
    psf_mirror = psf[::1, ::-1, ::-1]

    for i in range(iterations):
        print(f"iter {i+1}")
        relative_blur = image / convolve(im_deconv, psf, mode="same")
        im_deconv *= convolve(relative_blur, psf_mirror, mode="same")

    if clip:
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < -1] = -1

    return im_deconv


def main(image_path, psf_path):
    image = imageio.volread(image_path)  # z 0.6u
    psf = imageio.volread(psf_path)  # z 0.2u

    image, psf = image.astype(np.float32), psf.astype(np.float32)

    # crop image (too large)
    image = image[:, 896 : 896 + 256, 896 : 896 + 256]

    # rescale psf
    psf = rescale(psf, (1 / 3, 1, 1), anti_aliasing=False)
    # normalize
    psf /= psf.sum()

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
