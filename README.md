# Shape from Shading (SfS)

This module implements the Shape from Shading Algorithm using first principles.

### Assumptions:
- Orthographic Projection; Camera far away from object
- Source direction known
- Lambertian body 
- Reflectance model R = (n.s)^alpha

## Steps:
- Generate Image
- Detect Occluding Boundary
- Compute normal at occluding boundary
- Compute f,g from normals at occluding boundary
- Solve minimisation problem with f,g on boundary as initial condition
- Convert f,g to p,q 
- Integrate p,q to get depth map


## Directory Structure
```bash
├── README.md
├── coil-20-proc #Image Library
├── EE702_Assignment1.pdf
├── edge_detection.py
├── gen_reflectance.py
├── minimisation.py
├── depth_reconstruction.py
├── main.py
├── DepthMap2D.png
├── DepthMap3D.png
├── DetectedEdges.png
├── InitialCondition_fg.png
├── Normals_OccludingBoundary.png
├── shaded_sphere.png
├── LICENSE
├── .gitignore
```

## Installation
```bash
git clone https://github.com/your-username/your-project.git
cd your-project
```

## Usage
Load image by modifying the following part in main.py
```python
# LOADING IMAGES
# Load image from a folder in the working directory
image_path = "coil-20-proc/obj1__18.png"  # Replace with the actual folder and file name

```

```bash
python main.py
```

## Contributing
Contributions are welcome! If you find issues or have suggestions, please open an issue or submit a pull request. When contributing:
- Follow the existing coding style
- Update comments and documentation as necessary

## License
This project is licensed under the MIT License -- See the LICENSE file for details.

## Acknowledgements
- Prof Subhasis Chaudhuri, Instructor EE702, IIT Bombay
- [First Principles of Computer Vision](https://fpcv.cs.columbia.edu/) : Lecture Series by Prof Shree Nayar, Columbia University
- [Columbia Object Image Library (COIL-20)](https://cave.cs.columbia.edu/repository/COIL-20)
- ChatGPT has been used to help generate parts of this code and documentation


