import logging
import numpy as np
from skimage.transform import rescale

__all__ = ["cuboid_net"]

logger = logging.getLogger("utoolbox.util.preview")


def _generate_net_faces(array, scale=(1, 1, 1)):
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
        scale (tuple of float, optional): normalized scale
    """

    # array slicing -> actual ax -> net image
    #   0 XZ, ZX, [::-1, 0, :]
    #   1 YZ, YZ, [:, :, 0]
    #   2 XY, YX, [0, :, :]
    #   3 YZ, YZ, [:, ::-1, 0]
    #   4 XY, YX, [-1, :, ::-1]
    #   5 XZ, ZX, [:, -1, :]

    faces = [
        array[::-1, 0, :],
        np.rot90(np.array(array[:, :, 0]), k=1, axes=(1, 0)),
        array[0, :, :],
        np.rot90(np.array(array[::-1, :, -1]), k=1, axes=(1, 0)),
        array[-1, :, ::-1],
        array[:, -1, :],
    ]

    # scale each face to their correct aspect ratio
    sz, sy, sx = scale
    ratio = [(sz, sx), (sy, sz), (sy, sx), (sy, sz), (sy, sx), (sz, sx)]
    for i, (scale, face) in enumerate(zip(ratio, faces)):
        faces[i] = rescale(face, scale, anti_aliasing=True, preserve_range=True)

    return faces


def cuboid_net(array, scale=(1, 1, 1), gap=1, return_faces=False):
    """
    Generate the net with optimal output size.

    Args:
        array (np.array): a 3-D stack
        scale (tuple of float): voxel scale of the stack, default to isotropic
        gap (int, optional): gap between faces
        return_faces (bool, optional): return the raw faces
    """
    dtype = array.dtype
    
    #       +---+
    #       | 0 |
    #   +---+---+---+---+
    #   | 1 | 2 | 3 | 4 |
    #   +---+---+---+---+
    #       | 5 |
    #       +---+
    faces0 = _generate_net_faces(array, scale)
    nz, ny, nx = faces0[0].shape[0], faces0[2].shape[0], faces0[2].shape[1]

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
