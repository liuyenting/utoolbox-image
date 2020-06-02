import logging

import click
import numpy as np

from utoolbox.io import open_dataset
from utoolbox.io.dataset import (
    MultiChannelDatasetIterator,
    MultiViewDatasetIterator,
    TiledDatasetIterator,
    TimeSeriesDatasetIterator,
)

__all__ = ["preview"]

logger = logging.getLogger("utoolbox.cli.dataset")


@click.group()
@click.pass_context
def preview(ctx):
    """Generate previews for the dataset."""
    pass


@preview.command()
@click.argument("path")
@click.pass_context
def mip(ctx, path):
    """Generate maximum intensity projections."""
    raise NotImplementedError


@preview.command()
@click.argument("path")
@click.pass_context
def surface(ctx, path):
    """Extract data chunk surface layers."""
    show_trace = logger.getEffectiveLevel() <= logging.DEBUG
    src_ds = open_dataset(path, show_trace=show_trace)

    for time, _ in TimeSeriesDatasetIterator(src_ds):
        if time is None:
            break
        else:
            raise ValueError("surface preview does not support time series dataset")

    # IJ hyperstack order T[Z][C]YXS
    # re-purpose axis meaning:
    #   - Z, slice
    #   - C, channel
    iterator = TiledDatasetIterator(
        src_ds, axes="zyx", return_key=True, return_format="index"
    )
    for tile, t_ds in iterator:
        tile = "_".join(f"{ax:03d}" for ax in zip("xyz", reversed(tile)))

        for view, v_ds in MultiViewDatasetIterator(t_ds):
            for channel, c_ds in MultiChannelDatasetIterator(v_ds):
                print(f"{tile}, {view}, {channel}")
                print(src_ds[c_ds])


def _generate_net_faces(array):
    """
    Generate each face of the cuboid.

          +---+               +---+
          |XZ |               | 0 |
      +---+---+---+---+   +---+---+---+---+
      |YZ |XY |YZ |XY |   | 1 | 2 | 3 | 4 |
      +---+---+---+---+   +---+---+---+---+
          |XZ |               | 5 |            
          +---+               +---+

    Args:
        array (np.ndarray): a 3-D stack
    """

    # array slicing -> net image
    #   0 XZ [::-1, 0, :]
    #   1 YZ [:, :, 0]
    #   2 XY [0, :, :]
    #   3 YZ [:, ::-1, 0]
    #   4 XY [-1, :, ::-1]
    #   5 XZ [:, -1, :]

    return [
        array[::-1, 0, :],
        np.rot90(array[:, :, 0], k=1, axes=(1, 0)),
        array[0, :, :],
        np.rot90(array[::-1, :, -1], k=1, axes=(1, 0)),
        array[-1, :, ::-1],
        array[:, -1, :],
    ]


def generate_net(array, gap=1, return_faces=False):
    """
    Generate the net with optimal output size.

    Args:
        array (np.array): a 3-D stack
        gap (int, optional): gap between faces
        return_faces (bool, optional): return the raw faces
    """
    dtype, (nz, ny, nx) = array.dtype, array.shape

    # 3 types of net (in order)
    #
    #       +---+
    # Z     |XZ |
    #   +---+---+---+---+
    # Y |YZ |XY |YZ |XY |
    #   +---+---+---+---+
    # Z     |XZ |
    #       +---+
    #    Y   X   Y   X
    #
    #       +---+
    # Y     |XY |
    #   +---+---+---+---+
    # Z |YZ |XZ |YZ |XZ |
    #   +---+---+---+---+
    # Y     |XY |
    #       +---+
    #    Y   X   Y   X
    #
    #       +---+
    # X     |YX |
    #   +---+---+---+---+
    # Z |XZ |YZ |XZ |YZ |
    #   +---+---+---+---+
    # X     |YX |
    #       +---+
    #    X   Y   X   Y
    #

    # calculate blank area
    a1 = (nx + 2 * ny) * nz
    a2 = (nx + 2 * ny) * ny
    a3 = (2 * nx + ny) * nx

    # choose the one whose final X/Y is closer to 1
    ai = np.array([a1, a2, a3])
    ai = np.argmin(ai)

    logger.debug(f"using type {ai} net layout")

    # Type 0, XY
    #       +---+
    #       | 0 |
    #   +---+---+---+---+
    #   | 1 | 2 | 3 | 4 |
    #   +---+---+---+---+
    #       | 5 |
    #       +---+
    #
    # Type 1, XZ
    #       +---+
    #       | 2 |
    #   +---+---+---+---+
    #   |1L | 5 |3R |0RR|
    #   +---+---+---+---+
    #       |4RR|
    #       +---+
    #
    # Type 3, YZ
    #       +---+
    #       |0L |
    #   +---+---+---+---+
    #   | 4 | 1 | 2 | 3 |
    #   +---+---+---+---+
    #       |5R |
    #       +---+
    #
    #   L(+): left
    #   R(-): right
    ind_lut = [[0, 1, 2, 3, 4, 5], [2, 1, 5, 3, 0, 4], [0, 4, 1, 2, 3, 5]]
    rot_lut = [[0, 0, 0, 0, 0, 0], [0, -1, 0, 1, 2, 2], [-1, 0, 0, 0, 0, 1]]

    faces0 = _generate_net_faces(array)
    faces = []
    for ind, rot in zip(ind_lut[ai], rot_lut[ai]):
        face = faces0[ind]
        face = np.rot90(face, k=rot, axes=(1, 0))
        faces.append(face)

    # generate canvas
    nx = 2 * (faces[1].shape[1] + faces[2].shape[1]) + 3 * gap
    ny = 2 * faces[0].shape[0] + faces[2].shape[0] + 2 * gap
    canvas = np.zeros((ny, nx), dtype)

    # place faces onto the canvas
    offsets = [
        (0, faces[1].shape[1] + gap),
        (faces[0].shape[0] + gap, 0),
        (faces[0].shape[0] + gap, faces[1].shape[1] + gap),
        (faces[0].shape[0] + gap, faces[1].shape[1] + faces[2].shape[1] + 2 * gap),
        (
            faces[0].shape[0] + gap,
            faces[1].shape[1] + faces[2].shape[1] + faces[3].shape[1] + 3 * gap,
        ),
        (faces[0].shape[0] + faces[1].shape[0] + 2 * gap, faces[1].shape[1] + gap),
    ]
    for (oy, ox), face in zip(offsets, faces):
        ny, nx = face.shape
        canvas[oy : oy + ny, ox : ox + nx] = face

    if return_faces:
        return canvas, faces0
    else:
        return canvas


if __name__ == "__main__":
    import imageio

    data = imageio.volread(
        "kidney_Iter_0_ch1_stack0000_488nm_0000000msec_0075013047msecAbs.tif"
    )
    canvas = generate_net(data)
    imageio.imwrite("canvas.tif", canvas)
    # faces = _generate_net_faces(data)
    # for i, image in enumerate(faces):
    #    imageio.imwrite(f"{i:02d}.tif", image)
