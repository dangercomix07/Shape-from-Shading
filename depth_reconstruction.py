import numpy as np
import matplotlib.pyplot as plt
from edge_detection import compute_fg_edges
from generate_img import generate_image
from gen_reflectance_map import compute_reflectance
from minimisation import update_fg

# Integrate Gradients to Recover Depth
def integrate_depth(f, g, mask, z_gt, scale = False, save=True, visual=False):
    # Convert to p,q
    p = np.zeros_like(f)
    q = np.zeros_like(g)

    denominator = (4 - (f**2 + g**2))
    p[mask] = 4*f[mask]/denominator[mask]
    q[mask] = 4*g[mask]/denominator[mask]

    z = np.zeros_like(p)

    # Step size based on image grid resolution
    dx = 2 / (p.shape[1] - 1)
    dy = 2 / (q.shape[0] - 1)
    
    for i in range(1, z.shape[1]): # Integrate along x-axis (row-wise cumulative sum)
        z[:, i] = z[:, i-1] + p[:, i] * dx
    for j in range(1, z.shape[0]): # Integrate along y-axis (column-wise cumulative sum) in reversed order
        z[j, :] = z[j-1, :] + q[j, :] * dy # for integration along +y cartesian

    if scale == True:
        scale_factor = 1.0/np.max(z)
        z = z*scale_factor
        print("Scale factor applied:", scale_factor)
        print("Max calibrated depth:", np.nanmax(z))
    else:
        pass

    if save == True:
        # 3D Depth Visualization - Estimated vs Ground Truth
        x = np.linspace(-2, 2, z.shape[1])
        y = np.linspace(-2, 2, z.shape[0])
        X, Y = np.meshgrid(x, y)

        plt.figure(figsize=(6, 6))
        plt.imshow(z, cmap='jet', extent=(-1, 1, -1, 1))
        plt.colorbar()
        plt.title("Depth-Map")
        plt.savefig('DepthMap2D.png', dpi=300)

        # fig = plt.figure(figsize=(12, 6))
        # ax1 = fig.add_subplot(121, projection='3d')
        # ax1.plot_surface(X, Y, z, cmap='viridis', edgecolor='none')
        # ax1.set_title('3D Estimated Depth')
        # ax1.set_xlabel('X')
        # ax1.set_ylabel('Y')
        # ax1.set_zlabel('Depth')
        # set_axes_equal(ax1)  # Apply equal scaling

        # ax2 = fig.add_subplot(122, projection='3d')
        # ax2.plot_surface(X, Y, z_gt, cmap='viridis', edgecolor='none')
        # ax2.set_title('3D Ground Truth Depth')
        # ax2.set_xlabel('X')
        # ax2.set_ylabel('Y')
        # ax2.set_zlabel('Depth')
        # set_axes_equal(ax2)  # Apply equal scaling

        fig = plt.figure(figsize=(6, 6))
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.plot_surface(X, Y, z, cmap='viridis', edgecolor='none')
        ax1.set_title('3D Estimated Depth')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Depth')
        set_axes_equal(ax1)  # Apply equal scaling

        plt.savefig('DepthMap3D.png', dpi=300)
        if visual == True:
            plt.pause(0.001)
            return z
        else:
            return z
    else:
        return z

def set_axes_equal(ax):
    """Set equal scale for 3D plot axes."""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d()
    ])
    center = limits.mean(axis=1)
    max_range = (limits[:, 1] - limits[:, 0]).max() / 2

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

if __name__ == "__main__":
    E, normals_gt, z_gt = generate_image(64, save=False, return_depth=True, s_perturb=False)
    f, g, N_vectors, mask = compute_fg_edges(E)
    lambda_reg = 100.0
    f, g = update_fg(f, g, E, mask, lambda_reg,max_iterations=100,tol=1e-3)
    z = integrate_depth(f,g, mask)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(z, cmap='jet', extent=(-1, 1, -1, 1))
    plt.colorbar()
    plt.title("Depth-Map")
    
    # 3D Depth Visualization - Estimated vs Ground Truth
    x = np.linspace(-2, 2, z.shape[1])
    y = np.linspace(-2, 2, z.shape[0])
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, z, cmap='viridis', edgecolor='none')
    ax1.set_title('3D Estimated Depth')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Depth')
    set_axes_equal(ax1)  # Apply equal scaling

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, z_gt, cmap='viridis', edgecolor='none')
    ax2.set_title('3D Ground Truth Depth')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Depth')
    set_axes_equal(ax2)  # Apply equal scaling

    plt.savefig('DepthMap3D.png', dpi=300)
    plt.show()
