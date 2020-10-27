#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import numpy as np

from pyquaternion import Quaternion

def main():
    parser = argparse.ArgumentParser(description="Run stereo training loop.")
    parser.add_argument("pose_file", help="Path to gta poses file.")
    parser.add_argument("output_file", help="Path to output colmap file.")
    args = parser.parse_args()

    camera_id = 0

    # Load gta_sfm poses.
    pose_file = args.pose_file

    pose_data = np.loadtxt(pose_file, skiprows=1, dtype=np.float32)
    pose_ids = pose_data[:, 0]
    poses = pose_data[:, 1:]

    # Image list with two lines of data per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    output_file = args.output_file
    with open(output_file, "w") as target:
        target.write("# Number of images: {}\n".format(len(pose_ids)))

        for pose_idx in range(len(pose_ids)):
            cam_in_world = poses[pose_idx, :].reshape(4, 4)

            world_in_cam = np.linalg.inv(cam_in_world)
            R_world_in_cam = world_in_cam[:3, :3]
            t_world_in_cam = world_in_cam[:3, 3]

            q_world_in_cam = Quaternion(matrix=R_world_in_cam, atol=1e-3, rtol=1e-3)

            target.write("{} {} {} {} {} {} {} {} {} {:06d}.jpg\n".format(
                pose_idx,
                q_world_in_cam.w, q_world_in_cam.x, q_world_in_cam.y, q_world_in_cam.z,
                t_world_in_cam[0], t_world_in_cam[1], t_world_in_cam[2],
                camera_id, pose_idx))
            target.write("\n")
    
    return

if __name__ == '__main__':
    main()
