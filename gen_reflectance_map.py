import numpy as np
import matplotlib.pyplot as plt
from edge_detection import compute_fg_edges
from generate_img import generate_image

def compute_reflectance(f, g, mask):
    """
    Compute reflectance values from f, g using Lambertian reflectance model.
    
    :param f: Stereographic gradient f
    :param g: Stereographic gradient g
    :param light_source: Light source direction as (sx, sy, sz)
    :return: Reflectance values R(f, g)
    """
    sx, sy, sz = [0,0,1]
    R = np.zeros_like(f)

    # Compute normal components from f, g
    
    Nx = 4 * f / (4 + (f**2 + g**2))
    Ny = 4 * g / (4 + (f**2 + g**2))
    Nz = (4 -(f**2 + g**2)) / (4 + (f**2 + g**2))

    # Compute reflectance using Lambertian model
    R[mask] = sx * Nx[mask] + sy * Ny[mask] + sz * Nz[mask]
    R[~mask] = np.nan
    
    # Ensure non-negative reflectance (clamping to avoid negative lighting)
    R = np.maximum(R, 0)
    
    return R

if __name__ == "__main__":

    E, normals_gt, z_gt = generate_image(248,save=False, return_depth=True,s_perturb = False)
    f, g, N_vectors, mask = compute_fg_edges(E)

    # Compute reflectance
    R = compute_reflectance(f, g, mask)
    
    # Plot reflectance map
    plt.figure(figsize=(6, 6))
    plt.imshow(R, cmap='gray', extent=(-1, 1, -1, 1))
    plt.colorbar()
    plt.title("Reflectance Map R(f, g)")
    plt.show()