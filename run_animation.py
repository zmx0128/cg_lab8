import os
import argparse
import imageio

from run_lbs_lab import (
    install_chumpy_pickle_shim,
    make_out_dir,
    resolve_script_path,
    to_numpy,
    draw_mesh,
    compute_manual_lbs,
    build_demo_pose,
    build_demo_shape,
)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import smplx


def main(args):
    device = torch.device("cpu")
    dtype = torch.float32
    model_dir = resolve_script_path(args.model_dir)
    out_dir = resolve_script_path(args.out_dir)
    make_out_dir(out_dir)
    
    frames_dir = os.path.join(out_dir, "frames")
    make_out_dir(frames_dir)
    
    install_chumpy_pickle_shim()
    model = smplx.create(
        model_path=model_dir,
        model_type="smpl",
        gender="neutral",
        ext="pkl",
        num_betas=args.num_betas,
    ).to(device)
    
    faces = np.asarray(model.faces, dtype=np.int32)
    num_frames = args.num_frames
    
    betas = torch.zeros((1, args.num_betas), dtype=dtype, device=device)
    
    joint_names = {
        "left_hip": 1,
        "right_hip": 2,
        "left_knee": 4,
        "right_knee": 5,
        "left_shoulder": 16,
        "right_shoulder": 17,
        "left_elbow": 18,
        "right_elbow": 19,
    }
    
    target_joint = "right_elbow"
    target_joint_id = joint_names[target_joint]
    start_angle = 0.0
    end_angle = 1.5
    
    frames = []
    
    for frame_idx in range(num_frames):
        progress = frame_idx / (num_frames - 1)
        current_angle = start_angle + progress * (end_angle - start_angle)
        
        global_orient = torch.zeros((1, 3), dtype=dtype, device=device)
        body_pose = torch.zeros((1, 23 * 3), dtype=dtype, device=device)
        
        start = (target_joint_id - 1) * 3
        body_pose[0, start + 0] = current_angle
        
        data = compute_manual_lbs(model, betas, global_orient, body_pose)
        
        weight_scalar = to_numpy(model.lbs_weights[:, target_joint_id])
        
        fig = plt.figure(figsize=(6, 7))
        ax = fig.add_subplot(111, projection="3d")
        
        draw_mesh(
            ax,
            to_numpy(data["verts"][0]),
            faces,
            joints=to_numpy(data["J_transformed"][0]),
            vertex_scalar=weight_scalar,
            title=f"Frame {frame_idx+1}/{num_frames}\nRight Elbow: {current_angle:.2f} rad",
            elev=12,
            azim=108,
        )
        
        fig.tight_layout()
        
        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:03d}.png")
        fig.savefig(frame_path, dpi=150, bbox_inches="tight")
        
        frames.append(imageio.imread(frame_path))
        
        plt.close(fig)
        
        if (frame_idx + 1) % 10 == 0 or frame_idx == num_frames - 1:
            print(f"Generated frame {frame_idx + 1}/{num_frames}")
    
    gif_path = os.path.join(out_dir, "animation.gif")
    imageio.mimwrite(gif_path, frames, duration=1.0 / args.fps, loop=0)
    print(f"Saved GIF to {gif_path}")
    
    mp4_path = os.path.join(out_dir, "animation.mp4")
    try:
        imageio.mimwrite(mp4_path, frames, fps=args.fps)
        print(f"Saved MP4 to {mp4_path}")
    except Exception as e:
        print(f"Failed to save MP4: {e}")
    
    print(f"Animation complete! Results saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LBS Pose Animation")
    parser.add_argument("--model-dir", type=str, default="./models", help="模型目录")
    parser.add_argument("--out-dir", type=str, default="./outputs2", help="输出目录")
    parser.add_argument("--num-frames", type=int, default=60, help="动画帧数")
    parser.add_argument("--fps", type=int, default=30, help="帧率")
    parser.add_argument("--num-betas", type=int, default=10, help="shape 参数数量")
    args = parser.parse_args()
    main(args)
