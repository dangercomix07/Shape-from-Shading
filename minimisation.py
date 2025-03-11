import numpy as np
import matplotlib.pyplot as plt
from edge_detection import compute_fg_edges
from generate_img import generate_image
from gen_reflectance_map import compute_reflectance

def compute_dR_df_dg(f, g, mask, light_source=np.array([0, 0, 1])):
    """
    Compute partial derivatives dR/df and dR/dg for the reflectance function.
    
    :param f: Stereographic gradient f
    :param g: Stereographic gradient g
    :param light_source: Light source direction as (sx, sy, sz)
    :return: dR/df, dR/dg
    """
    sx, sy, sz = light_source

    denom = (4 + (f**2 + g**2))
    denom[denom < 1e-6] = 1e-3  # Avoid division by zero
    denom = np.maximum(denom, 1e-3)  # Prevent small denominator

    # Compute partial derivatives of normal components
    dNx_df = (4*denom - 8*f**2)/denom**2
    dNy_df = -8*f*g / denom**2
    dNz_df = -16*f/denom**2

    dNx_dg = -8*f*g / denom**2
    dNy_dg = (4*denom - 8*g**2)/denom**2
    dNz_dg = -16*g/denom**2

    # Compute dR/df and dR/dg
    dR_df = sx * dNx_df + sy * dNy_df + sz * dNz_df
    dR_dg = sx * dNx_dg + sy * dNy_dg + sz * dNz_dg

    return dR_df, dR_dg

def update_fg(f, g, E, mask, lambda_reg, max_iterations=100, tol=1e-6):
    """
    Iteratively update f, g using Jacobi method to minimize model error and smoothing error.
    
    :param f: Initial f matrix
    :param g: Initial g matrix
    :param E: Input intensity image
    :param lambda_reg: Regularization parameter
    :param max_iterations: Maximum number of iterations
    :param tol: Convergence threshold
    :return: Updated f, g matrices
    """
    valid_mask = ~np.isnan(f) & ~np.isnan(g)  # Ensure updates only in valid regions

    for k in range(max_iterations):
        R = compute_reflectance(f, g, mask)  # Compute reflectance map
        dR_df, dR_dg = compute_dR_df_dg(f, g ,mask)  # Compute derivatives
        
        f_new = np.copy(f)
        g_new = np.copy(g)
        
        # Iterate over valid surface region
        for i in range(1, f.shape[0] - 1):
            for j in range(1, f.shape[1] - 1):
                if valid_mask[i, j]:  # Only update inside valid region
                    local_avg_f = np.nanmean([f[i+1, j], f[i-1, j], f[i, j+1], f[i, j-1]])  # Ignore NaNs
                    local_avg_g = np.nanmean([g[i+1, j], g[i-1, j], g[i, j+1], g[i, j-1]])

                    # Ensure no NaN values in update step
                    local_avg_f = 0 if np.isnan(local_avg_f) else local_avg_f
                    local_avg_g = 0 if np.isnan(local_avg_g) else local_avg_g


                    update_f = float((1 / lambda_reg) * (E[i, j] - R[i, j]) * dR_df[i, j])
                    update_g = float((1 / lambda_reg) * (E[i, j] - R[i, j]) * dR_dg[i, j])

                    update_f = np.clip(update_f, -1, 1)
                    update_g = np.clip(update_g, -1, 1)
                    # Update rules
                    f_new[i, j] = local_avg_f + update_f
                    g_new[i, j] = local_avg_g + update_g
    
        max_f_change = np.nanmax(np.abs(f_new - f))
        max_g_change = np.nanmax(np.abs(g_new - g))
        print(f"Iteration {k}: max(abs(delta f)) = {max_f_change}, max(abs(delta g)) = {max_g_change}")

        # Check for convergence
        if 0.5*(np.nanmax(np.abs(f_new - f)) + np.nanmax(np.abs(g_new - g))) < tol:
            break
        #print("E - R min/max:", np.nanmin(E - R), np.nanmax(E - R))
        print("error",0.5*(np.nanmax(np.abs(f_new - f)) + np.nanmax(np.abs(g_new - g))))

        f, g = f_new, g_new
    
    return f, g

if __name__ == "__main__":
    E, normals_gt, z_gt = generate_image(64, save=True, return_depth=True, s_perturb=False)
    f, g, N_vectors, mask= compute_fg_edges(E)
    lambda_reg = 100.0
    
    f_updated, g_updated = update_fg(f, g, E, mask, lambda_reg,max_iterations=100,tol=1e-3)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(f_updated, cmap='jet')
    plt.colorbar()
    plt.title("Updated f")
    
    plt.subplot(1, 2, 2)
    plt.imshow(g_updated, cmap='jet')
    plt.colorbar()
    plt.title("Updated g")
    
    plt.show()
