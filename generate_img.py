import numpy as np
import matplotlib.pyplot as plt

# Define the surface (Sphere)
def sphere(x, y, r=1):
    z = np.sqrt(np.maximum(0, r**2 - x**2 - y**2))  # Avoid negative values
    return z

# Compute Surface Normals (Ideal - from known surface)
def compute_normals(x, y, z, r):
    mask = (x**2 + y**2) < r**2  # Valid region mask
    dzdx = np.zeros_like(x)
    dzdy = np.zeros_like(y)
    dzdx[mask] = -x[mask] / np.sqrt(r**2 - (x[mask]**2 + y[mask]**2))
    dzdy[mask] = -y[mask] / np.sqrt(r**2 - (x[mask]**2 + y[mask]**2))
    
    nz = np.zeros_like(z)
    nz[mask] = 1
    norm = np.sqrt(dzdx**2 + dzdy**2 + nz**2)
    n = np.stack((dzdx, dzdy, nz)) / (norm + 1e-10)  # Normalize
    return n

# Shading Model
def shading_model(normals, s, alpha=1):
    dot_product = np.tensordot(normals.T, s, axes=([2], [0]))  # n dot s
    dot_product = np.clip(dot_product, 0, 1)  # Remove negative values
    return dot_product ** alpha

# Function to Generate and Save Shaded Image & Normals
def generate_image(img_size = 64, save=True, return_depth=False, s_perturb =False):
    # Grid for X, Y coordinates
    x = np.linspace(-2, 2, img_size)
    y = np.linspace(-2, 2, img_size)
    x, y = np.meshgrid(x, y)
    r = 1
    z = sphere(x, y, r)
    
    # Compute Ideal Normals (from known surface)
    normals = compute_normals(x, y, z, r)

    if s_perturb == True:
        epsilon1 = np.random.uniform(-0.5, 0.5)
        epsilon2 = np.random.uniform(-0.5, 0.5)  
        s = np.array([epsilon1, epsilon2, 1])  # New perturbed light source direction
        s /= np.linalg.norm(s)  # Normalize
    else:
        s = np.array([0, 0, 1.00])
    
    #print(s)

    # Generate Shaded Image
    E = shading_model(normals, s, 1)
    mask = x**2 + y**2 > r**2
    E[mask] = 0  # Set outside the sphere to black
    
    if save:
        # Save Shaded Image and Normals as NumPy Files for Later Processing
        np.save('shaded_image.npy', E)
        #np.save('normals.npy', normals)
    
        # Save and Display Image
        plt.imshow(E, cmap='gray', extent=(-1, 1, -1, 1))
        plt.colorbar(label='Shading Intensity')
        plt.title('Shaded Image from Sphere')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('shaded_sphere.png', dpi=300)
        plt.pause(0.001)
    
    if return_depth:
        return E, normals, z
    return E, normals

# Run the function if script is executed standalone
if __name__ == "__main__":
    E, normals_gt, z = generate_image(save=True, return_depth=True)

    # # 3D Depth Visualization
    # x = np.linspace(-1, 1, z.shape[1])
    # y = np.linspace(-1, 1, z.shape[0])
    # X, Y = np.meshgrid(x, y)

    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, z, cmap='viridis', edgecolor='none')
    # ax.set_title('3D Depth Reconstruction')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Depth')
    # plt.savefig('depth_3d.png', dpi=300)
    # plt.show()