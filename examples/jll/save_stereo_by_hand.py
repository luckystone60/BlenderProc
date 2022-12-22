# blenderproc run examples/datasets/front_3d/save_stereo_by_hand.py /srv/02-data/02-3D_model/01-3DFront/3D-FRONT/c7454293-6e55-460a-accc-039ec046435b.json /srv/02-data/02-3D_model/01-3DFront/3D-FUTURE-model /srv/02-data/02-3D_model/01-3DFront/3D-FRONT-texture/ examples/datasets/front_3d/output_by_hand /srv/02-data/01-stereo_by_blender/01-scenes/c7454293-6e55-460a-accc-039ec046435b.txt

import blenderproc as bproc
from blenderproc.python.utility.SetupUtility import SetupUtility
from blenderproc.python.postprocessing.PostProcessingUtility import depth2disparity

import argparse
import os
import numpy as np
import mathutils

# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()

parser = argparse.ArgumentParser()
parser.add_argument("front", help="Path to the 3D front file")
parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
parser.add_argument("front_3D_texture_path", help="Path to the 3D FRONT texture folder.")
parser.add_argument("output_dir", help="Path to where the data should be saved")
parser.add_argument('camera', help="Path to the camera file, should be examples/resources/camera_positions")
args = parser.parse_args()

if not os.path.exists(args.front) or not os.path.exists(args.future_folder):
    raise Exception("One of the two folders does not exist!")

bproc.init()
mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

# set the light bounces
bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                  transmission_bounces=200, transparent_max_bounces=200)

# load the front 3D objects
loaded_objects = bproc.loader.load_front3d(
    json_path=args.front,
    future_model_path=args.future_folder,
    front_3D_texture_path=args.front_3D_texture_path,
    label_mapping=mapping
)

# Init sampler for sampling locations inside the loaded front3D house
point_sampler = bproc.sampler.Front3DPointInRoomSampler(loaded_objects)

# Init bvh tree containing all mesh objects
bvh_tree = bproc.object.create_bvh_tree_multi_objects([o for o in loaded_objects if isinstance(o, bproc.types.MeshObject)])

def check_name(name):
    for category_name in ["chair", "sofa", "table", "bed"]:
        if category_name in name.lower():
            return True
    return False

# Set intrinsics via K matrix
bproc.camera.set_intrinsics_from_K_matrix(
    [[650.018, 0, 637.962],
    [0, 650.018, 355.984],
    [0, 0, 1]], 1280, 720
)

# Enable stereo mode and set baseline
bproc.camera.set_stereo_parameters(interocular_distance=0.01, convergence_mode="PARALLEL", convergence_distance=0)

# read the camera positions file and convert into homogeneous camera-world transformation
with open(args.camera, "r") as f:
    for line in f.readlines():
        line = [float(x) for x in line.split()]
        position, euler_rotation = line[:3], line[3:6]
        matrix_world = bproc.math.build_transformation_mat(position, euler_rotation)
        bproc.camera.add_camera_pose(matrix_world)

# Enable output type
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.material.add_alpha_channel_to_textures(blurry_edges=False)
bproc.renderer.toggle_stereo(True)

# Set max samples for quick rendering
bproc.renderer.set_max_amount_of_samples(15)

# render the whole pipeline
data = bproc.renderer.render()

data["disparity"] = depth2disparity(data["depth"])

# write the data to a .hdf5 container
bproc.writer.write_hdf5(args.output_dir, data)

# write the animations into .gif files
bproc.writer.write_gif_animation(args.output_dir, data, frame_duration_in_ms=80)