import torch
import numpy as np


def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents, weights, pose2rot=True):
    batch_size = max(betas.shape[0], pose.shape[0])
    
    v_shaped = v_template + blend_shapes(betas, shapedirs)
    J = vertices2joints(J_regressor, v_shaped)
    
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(batch_size, -1, 3, 3)
    else:
        rot_mats = pose
        
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents)
    v_posed = v_shaped + blend_shapes(pose[:, 3:].view(batch_size, -1, 3), posedirs)
    v_homogeneous = torch.cat([v_posed, torch.ones([batch_size, v_posed.shape[1], 1])], dim=-1)
    
    T = torch.bmm(weights, A.view(batch_size, -1, 16)).view(batch_size, -1, 4, 4)
    v_homogeneous = v_homogeneous.unsqueeze(-1)
    v_deformed = torch.matmul(T, v_homogeneous).squeeze(-1)[:, :, :3]
    
    return v_deformed, J_transformed


def blend_shapes(betas, shape_disps):
    blend_weights = betas[:, None, :]
    blend_shape = torch.matmul(blend_weights, shape_disps)
    return blend_shape


def vertices2joints(J_regressor, vertices):
    J = torch.einsum('bik, kj->bij', J_regressor, vertices)
    return J


def batch_rodrigues(rot_vecs):
    batch_size = rot_vecs.shape[0]
    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle
    
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    
    outer = torch.bmm(rot_dir.unsqueeze(-1), rot_dir.unsqueeze(1))
    
    diag = torch.eye(3, device=rot_vecs.device).unsqueeze(0).repeat(batch_size, 1, 1)
    diag = diag - outer
    
    R = cos.unsqueeze(-1) * diag + sin.unsqueeze(-1) * hat(rot_dir) + outer
    
    return R


def hat(v):
    batch_size = v.shape[0]
    return torch.stack([
        torch.zeros(batch_size, 3, device=v.device),
        -v[:, 2], v[:, 1],
        v[:, 2], torch.zeros(batch_size, device=v.device), -v[:, 0],
        -v[:, 1], v[:, 0], torch.zeros(batch_size, device=v.device)
    ], dim=1).view(batch_size, 3, 3)


def batch_rigid_transform(rot_mats, joints, parents):
    batch_size = rot_mats.shape[0]
    num_joints = joints.shape[1]
    
    J_transformed = torch.zeros_like(joints)
    A = torch.zeros(batch_size, num_joints, 4, 4, device=rot_mats.device)
    
    J_transformed[:, 0] = joints[:, 0]
    A[:, 0, :3, :3] = rot_mats[:, 0]
    A[:, 0, :3, 3] = J_transformed[:, 0]
    A[:, 0, 3, 3] = 1.0
    
    for i in range(1, num_joints):
        J_transformed[:, i] = torch.matmul(rot_mats[:, i-1], (joints[:, i] - joints[:, parents[i]]).unsqueeze(-1)).squeeze(-1) + J_transformed[:, parents[i]]
        A[:, i, :3, :3] = torch.matmul(A[:, parents[i], :3, :3], rot_mats[:, i])
        A[:, i, :3, 3] = J_transformed[:, i]
        A[:, i, 3, 3] = 1.0
    
    return J_transformed, A


def compute_vertex_weights(weights, joint_id):
    return weights[:, :, joint_id]


def visualize_weight_heatmap(vertices, weights, joint_id, ax):
    weights_np = weights.detach().cpu().numpy()[0, :, joint_id]
    vertices_np = vertices.detach().cpu().numpy()[0]
    
    scatter = ax.scatter(vertices_np[:, 0], vertices_np[:, 1], c=weights_np, cmap='viridis', s=1)
    ax.set_title(f'Joint {joint_id} Weight Heatmap')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    return scatter
