import time

import imageio
import numpy as np
from pyotf.phaseretrieval import retrieve_phase
from pyotf.utils import prep_data_for_PR
import matplotlib.pyplot as plt

# read in data from fixtures
def main(psf_path):
    data = imageio.volread(psf_path)

    # re-center the data
    cz, cy, cx = np.unravel_index(np.argmax(data), shape=data.shape)
    nz, ny, nx = data.shape

    data = [
        data[cz - 20, ...],
        data[cz - 15, ...],
        data[cz - 10, ...],
        data[cz - 5, ...],
        data[cz, ...],
        data[cz + 5, ...],
        data[cz + 10, ...],
        data[cz + 15, ...],
        data[cz + 20, ...],
    ]  # 200 per layer, 2 layer each
    data = np.array(data)

    # prep data
    data_prepped = prep_data_for_PR(data, 256)

    # set up model params
    params = dict(wl=520, na=1.05, ni=1.33, res=102, zres=1000)

    # retrieve the phase
    pr_start = time.time()
    print("Starting phase retrieval ... ", end="", flush=True)
    pr_result = retrieve_phase(data_prepped, params, max_iters=100)
    pr_time = time.time() - pr_start
    print(f"{pr_time:.1f} seconds were required to retrieve the pupil function")

    # plot
    pr_result.plot()
    pr_result.plot_convergence()

    # fit to zernikes
    zd_start = time.time()
    print("Starting zernike decomposition ... ", end="", flush=True)
    pr_result.fit_to_zernikes(120)
    zd_time = time.time() - zd_start
    print(f"{zd_time:.1f} seconds were required to fit 120 Zernikes")

    # plot
    pr_result.zd_result.plot_named_coefs()
    pr_result.zd_result.plot_coefs()

    plt.show()

    # save as tiff
    print("Write back...")
    zrange = np.linspace(-(nz - 1) // 2, (nz - 1) // 2, nz)
    zrange *= 200
    psf = pr_result.generate_psf(size=nx, zsize=200, zrange=zrange)

    imageio.volwrite("psf_hanser_pr.tif", psf.astype(np.float32))


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    main("C:/Users/Andy/Desktop/background_removal/psf/NA1p05_zp2um_cropped.tif")
