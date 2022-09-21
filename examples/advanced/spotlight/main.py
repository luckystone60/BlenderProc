import blenderproc as bproc
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('scene', nargs='?', default="examples/resources/scene.obj", help="Path to the scene.obj file")
parser.add_argument('output_dir', nargs='?', default="examples/advanced/spotlight/output",
                    help="Path to where the final files, will be saved")
args = parser.parse_args()

bproc.init()

# load the objects into the scene
objs = bproc.loader.load_obj(args.scene)

# Find point of interest, all cam poses should look towards it
poi = bproc.object.compute_poi(objs)
# Sample five camera poses
for i in range(5):
    # Sample random camera location above objects
    location = np.random.uniform([-10, -10, 8], [10, 10, 12])
    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location,
                                                             inplane_rot=np.random.uniform(-0.7854, 0.7854))
    # Add homog cam pose based on location a rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)

# add a spotlight which intersect with the sampled camera poses
spot_light = bproc.lighting.add_intersecting_spot_lights_to_camera_poses(clip_start=10, clip_end=15)

# activate normal and depth rendering
bproc.renderer.enable_normals_output()
bproc.renderer.enable_depth_output(activate_antialiasing=False)

# render the whole pipeline
data = bproc.renderer.render()

# write the data to a .hdf5 container
bproc.writer.write_hdf5(args.output_dir, data)
