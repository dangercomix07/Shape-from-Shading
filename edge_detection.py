import numpy as np
import cv2
import matplotlib.pyplot as plt
from generate_img import generate_image

def compute_fg_edges(E,return_edges=False,save=False):
    """
    Compute f, g values at image edges using occluding boundary conditions.
    :param E: Input intensity image
    :return: f, g values at edges
    """
    # Normalize image if not already uint8
    if E.dtype != np.uint8:
        Enorm = cv2.normalize(E, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        Enorm = E.copy()
    
    # Edge detection using Canny
    edge_vals = cv2.Canny(Enorm, 120, 150)
    
    # Compute image gradients
    grad_x = cv2.Sobel(Enorm, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = -cv2.Sobel(Enorm, cv2.CV_64F, 0, 1, ksize=5)
    
    # Initialize f, g matrices
    f = np.zeros_like(E, dtype=np.float64)
    g = np.zeros_like(E, dtype=np.float64)
    N_vectors = np.zeros((E.shape[0], E.shape[1], 3), dtype=np.float64)

    # Viewing direction vector
    V = np.array([0, 0, 1])
    
    # Find edge pixel coordinates
    edges = np.column_stack(np.where(edge_vals > 0))
    
    # Compute f, g at edge points
    for y, x in edges:
        gx, gy = grad_x[y, x], grad_y[y, x]
        
        if gx**2 + gy**2 > 1e-6:  # Avoid division by zero
            # Compute tangent vector at the edge
            tangent = np.array([gy, -gx, 0])  # Tangent in 3D space (z=0)
            tangent /= np.linalg.norm(tangent)  # Normalize tangent
            
            # Compute normal by cross product with viewing direction
            N_vector = np.cross(tangent, V)
            N_vector /= np.linalg.norm(N_vector)  # Normalize normal
            N_vectors[y, x] = N_vector  # Store normal vector

            # Compute f, g directly from normal to avoid singularities
            f[y, x] = 2 * N_vector[0] / (1 + N_vector[2])
            g[y, x] = 2 * N_vector[1] / (1 + N_vector[2])

    # Create mask for valid surface region (boundary + inside)
    mask = np.zeros_like(E, dtype=np.uint8)
    
    # Find contours (detect occluding boundary)
    contours, _ = cv2.findContours(edge_vals, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, 1, thickness=cv2.FILLED)  # Fill inside contour
    mask = mask.astype(bool)

    # Apply mask to f and g (set values outside surface to NaN)
    f[~mask] = np.nan
    g[~mask] = np.nan

    if save == False:
        if return_edges:
            return f,g, N_vectors, edges ,mask
        else:
            return f, g, N_vectors, mask
    else:
        E_uint8 = cv2.normalize(E, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # Convert grayscale image to RGB for color overlay
        overlay = cv2.cvtColor(E_uint8, cv2.COLOR_GRAY2BGR)

        # # Ensure edge coordinates are valid before overlaying
        # valid_edges = (edges[:, 0] < E.shape[0]) & (edges[:, 1] < E.shape[1])
        # y_valid, x_valid = edges[valid_edges, 0], edges[valid_edges, 1]

        [x,y] = edges[:, 0], edges[:, 1]
        # Set detected edges to red
        overlay[y, x] = (255, 0, 0)  # Red color for edges

        plt.figure(figsize=(6, 6))
        plt.imshow(overlay, cmap='gray', extent=(-1, 1, -1, 1))
        plt.title("Detected Edges (Canny)")
        plt.axis("off")

        # Plot results
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(f, cmap='jet', extent=(-1, 1, -1, 1))
        plt.colorbar()
        plt.title("f values at edges")
        
        plt.subplot(1, 2, 2)
        plt.imshow(g, cmap='jet', extent=(-1, 1, -1, 1))
        plt.colorbar()
        plt.title("g values at edges")
        plt.savefig('InitalCondition_fg.png', dpi=300)

        # Plot normals as quiver plot
        plt.figure(figsize=(6, 6))
        plt.imshow(E, cmap='gray')
        plt.title("Normal Vectors at Edges")
        y_coords, x_coords = np.where(np.linalg.norm(N_vectors, axis=2) > 0)  # Find valid normal vectors
        plt.quiver(x_coords, y_coords, N_vectors[y_coords, x_coords, 0], -N_vectors[y_coords, x_coords, 1], color='red', angles='xy', scale_units='xy', scale=1)
        plt.savefig('Normals_OccludingBoundary.png', dpi=300)

        plt.pause(0.001)

        if return_edges:
            return f,g, N_vectors, edges ,mask
        else:
            return f, g, N_vectors, mask
        
if __name__ == "__main__":
    # Example: Load or generate synthetic image
    # Load Shaded Image and Ground Truth Normals
    E, normals_gt, z_gt = generate_image(64,save=False, return_depth=True,s_perturb = False)
    f, g, N_vectors, edges, mask = compute_fg_edges(E,return_edges=True,save=True)

    # E_uint8 = cv2.normalize(E, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # # Convert grayscale image to RGB for color overlay
    # overlay = cv2.cvtColor(E_uint8, cv2.COLOR_GRAY2BGR)

    # # # Ensure edge coordinates are valid before overlaying
    # # valid_edges = (edges[:, 0] < E.shape[0]) & (edges[:, 1] < E.shape[1])
    # # y_valid, x_valid = edges[valid_edges, 0], edges[valid_edges, 1]

    # [x,y] = edges[:, 0], edges[:, 1]
    # # Set detected edges to red
    # overlay[y, x] = (255, 0, 0)  # Red color for edges

    # plt.figure(figsize=(6, 6))
    # plt.imshow(overlay, cmap='gray', extent=(-1, 1, -1, 1))
    # plt.title("Detected Edges (Canny)")
    # plt.axis("off")

    # # Plot results
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(f, cmap='jet', extent=(-1, 1, -1, 1))
    # plt.colorbar()
    # plt.title("f values at edges")
    
    # plt.subplot(1, 2, 2)
    # plt.imshow(g, cmap='jet', extent=(-1, 1, -1, 1))
    # plt.colorbar()
    # plt.title("g values at edges")

    # # Plot normals as quiver plot
    # plt.figure(figsize=(6, 6))
    # plt.imshow(E, cmap='gray')
    # plt.title("Normal Vectors at Edges")
    # y_coords, x_coords = np.where(np.linalg.norm(N_vectors, axis=2) > 0)  # Find valid normal vectors
    # plt.quiver(x_coords, y_coords, N_vectors[y_coords, x_coords, 0], -N_vectors[y_coords, x_coords, 1], color='red', angles='xy', scale_units='xy', scale=1)

    # # # Verify mask
    # # plt.imshow(mask, cmap='gray')
    # # plt.colorbar(label="Valid (1) / Invalid (0)")
    # # plt.title("Mask Verification (Valid Surface Pixels)")

    # plt.show()

