import torch
import numpy as np
import os
import sys
import re
from scipy.spatial.transform import Rotation as R

# ==========================================
# Helpers for Robust Data Loading & Shaping
# ==========================================

def load_pt_as_numpy(path):
    """Loads a .pt file safely and returns a numpy array."""
    if not os.path.exists(path):
        return None
    try:
        data = torch.load(path)
        if isinstance(data, torch.Tensor):
            return data.cpu().detach().numpy()
        return data
    except Exception as e:
        print(f"  [Error] Failed to load {path}: {e}")
        return None

def convert_quat_to_axis_angle(arr):
    """
    Converts (N, 4) Quaternions -> (N, 3) Axis-Angle.
    Arctic expects Axis-Angle for all rotations.
    """
    if arr.shape[-1] != 4:
        return arr # Assume it's already Axis-Angle or Rotation Matrix (handle elsewhere)
    
    try:
        # Scipy handles (x, y, z, w) or (w, x, y, z). 
        # Arctic/PyTorch usually uses (w, x, y, z) or (x, y, z, w).
        # We assume standard scalar-last (x, y, z, w) for now.
        rot = R.from_quat(arr)
        return rot.as_rotvec()
    except ValueError:
        print("  [Warning] Rotation conversion failed. Returning raw.")
        return arr[:, :3]

def normalize_shape_N_D(arr, target_dim=None, name="Array"):
    """
    Forces an array into (N, D) format.
    - Reshapes (N,) -> (N, 1)
    - Flattens (N, A, B) -> (N, A*B)
    - Squeezes (N, D, 1) -> (N, D)
    - Transposes (D, N) -> (N, D) if N >> D
    """
    if arr is None: 
        return None
    
    # 0. Handle 1D array: (N,) -> (N, 1)
    if arr.ndim == 1:
        arr = arr[:, None]

    # 1. Handle singleton trailing dims: (N, D, 1) -> (N, D)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr.squeeze(2)
    # 2. Handle singleton middle dims: (N, 1, D) -> (N, D)
    elif arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr.squeeze(1)
    
    # 3. Handle Transpose case: (3, N) -> (N, 3)
    # Heuristic: If dim 0 is small (<=4) and dim 1 is large (>10), assume transpose needed.
    if arr.ndim == 2 and arr.shape[0] <= 4 and arr.shape[1] > 10:
        print(f"  [Fix] Transposing {name} {arr.shape} -> ({arr.shape[1]}, {arr.shape[0]})")
        arr = arr.T

    # 4. Handle Flattening: (N, 15, 3) -> (N, 45)
    if arr.ndim > 2:
        N = arr.shape[0]
        D_flat = np.prod(arr.shape[1:])
        arr = arr.reshape(N, D_flat)

    # 5. Check Target Dimension
    if target_dim is not None:
        if arr.shape[1] != target_dim:
            # Special Handling: 4D Quat -> 3D AxisAngle
            if arr.shape[1] == 4 and target_dim == 3:
                print(f"  [Fix] Converting {name} Quaternions (4D) to Axis-Angle (3D)")
                arr = convert_quat_to_axis_angle(arr)
            # Special Handling: 48D Pose -> 3D Rot + 45D Pose (Caller should handle, but just in case)
            elif arr.shape[1] == 48 and target_dim == 45:
                 print(f"  [Fix] Trimming {name} 48D -> 45D (discarding first 3)")
                 arr = arr[:, 3:]
            elif arr.shape[1] < target_dim:
                print(f"  [Fix] Padding {name} {arr.shape} -> (N, {target_dim})")
                padding = np.zeros((arr.shape[0], target_dim - arr.shape[1]))
                arr = np.hstack([arr, padding])
            elif arr.shape[1] > target_dim:
                print(f"  [Fix] Clipping {name} {arr.shape} -> (N, {target_dim})")
                arr = arr[:, :target_dim]

    return arr

def synchronize_sequence_length(arr, target_len, name="Array"):
    """
    Ensures array has length `target_len`.
    - If N=1, broadcast to target_len.
    - If N < target, pad with last frame.
    - If N > target, clip.
    """
    if arr is None: 
        return None
        
    current_len = arr.shape[0]
    
    if current_len == target_len:
        return arr
        
    if current_len == 1:
        # Broadcast constant value
        return np.tile(arr, (target_len, 1))
        
    if current_len > target_len:
        # Clip
        return arr[:target_len]
        
    if current_len < target_len:
        # Pad with last frame
        diff = target_len - current_len
        print(f"  [Info] Padding {name} (len {current_len}) to match hand (len {target_len})")
        padding = np.tile(arr[-1:], (diff, 1))
        return np.vstack([arr, padding])

    return arr

def clean_seq_name(folder_name):
    name = folder_name
    name = re.sub(r'^s\d+_', '', name) # Remove s05_
    name = re.sub(r'_\d+$', '', name)  # Remove _0
    return name

# ==========================================
# Main Generator
# ==========================================

def generate_arctic_raw(source_root, output_dir=None, ref_smplx_path=None, ref_egocam_path=None):
    if output_dir is None:
        output_dir = source_root
    
    os.makedirs(output_dir, exist_ok=True)
    raw_folder_name = os.path.basename(os.path.normpath(source_root))
    seq_name = clean_seq_name(raw_folder_name)
    
    print(f"Processing: {raw_folder_name}")
    print(f"Output ID:  {seq_name}")
    print(f"Save Path:  {output_dir}")
    print("-" * 40)
    
    # ---------------------------------------------------------
    # 1. Load Hand Data (The "Master" Clock)
    # ---------------------------------------------------------
    # Load raw tensors
    pose_r = load_pt_as_numpy(os.path.join(source_root, 'preds/pred.mano.pose.r.pt'))
    rot_r  = load_pt_as_numpy(os.path.join(source_root, 'preds/pred.mano.rot.r.pt'))
    trans_r= load_pt_as_numpy(os.path.join(source_root, 'preds/pred.mano.cam_t.r.pt'))
    shape_r= load_pt_as_numpy(os.path.join(source_root, 'preds/pred.mano.beta.r.pt'))
    
    pose_l = load_pt_as_numpy(os.path.join(source_root, 'preds/pred.mano.pose.l.pt'))
    rot_l  = load_pt_as_numpy(os.path.join(source_root, 'preds/pred.mano.rot.l.pt'))
    trans_l= load_pt_as_numpy(os.path.join(source_root, 'preds/pred.mano.cam_t.l.pt'))
    shape_l= load_pt_as_numpy(os.path.join(source_root, 'preds/pred.mano.beta.l.pt'))

    # Logic: Handle combined pose/rot (48 dims)
    if pose_r is not None and pose_r.ndim == 2 and pose_r.shape[1] == 48:
        print("  [Info] Splitting Right Hand Pose (48) -> Rot(3) + Pose(45)")
        if rot_r is None: rot_r = pose_r[:, :3]
        pose_r = pose_r[:, 3:]
        
    if pose_l is not None and pose_l.ndim == 2 and pose_l.shape[1] == 48:
        print("  [Info] Splitting Left Hand Pose (48) -> Rot(3) + Pose(45)")
        if rot_l is None: rot_l = pose_l[:, :3]
        pose_l = pose_l[:, 3:]

    # Normalize Dimensions
    pose_r = normalize_shape_N_D(pose_r, target_dim=45, name="Right Pose")
    rot_r  = normalize_shape_N_D(rot_r,  target_dim=3,  name="Right Rot")
    trans_r= normalize_shape_N_D(trans_r, target_dim=3, name="Right Trans")
    
    pose_l = normalize_shape_N_D(pose_l, target_dim=45, name="Left Pose")
    rot_l  = normalize_shape_N_D(rot_l,  target_dim=3,  name="Left Rot")
    trans_l= normalize_shape_N_D(trans_l, target_dim=3, name="Left Trans")

    # Determine Master Sequence Length
    seq_len = 0
    if pose_r is not None: seq_len = len(pose_r)
    elif pose_l is not None: seq_len = len(pose_l)
    
    if seq_len == 0:
        print("[Error] No valid hand pose found. Exiting.")
        return
    print(f"  [Info] Master Sequence Length: {seq_len}")

    # Synchronize Hand Arrays (in case one side is missing or length differs)
    if pose_r is not None:
        pose_r = synchronize_sequence_length(pose_r, seq_len)
        rot_r = synchronize_sequence_length(rot_r, seq_len)
        trans_r = synchronize_sequence_length(trans_r, seq_len)
    if pose_l is not None:
        pose_l = synchronize_sequence_length(pose_l, seq_len)
        rot_l = synchronize_sequence_length(rot_l, seq_len)
        trans_l = synchronize_sequence_length(trans_l, seq_len)

    # ---------------------------------------------------------
    # 2. Save .mano.npy
    # ---------------------------------------------------------
    mano_data = {'left': {}, 'right': {}}
    
    if pose_r is not None:
        mano_data['right']['pose'] = pose_r
        mano_data['right']['rot'] = rot_r if rot_r is not None else np.zeros((seq_len, 3))
        mano_data['right']['trans'] = trans_r if trans_r is not None else np.zeros((seq_len, 3))
        mano_data['right']['shape'] = shape_r.flatten()[:10] if shape_r is not None else np.zeros(10)
        mano_data['right']['fitting_err'] = np.zeros(seq_len)
    
    if pose_l is not None:
        mano_data['left']['pose'] = pose_l
        mano_data['left']['rot'] = rot_l if rot_l is not None else np.zeros((seq_len, 3))
        mano_data['left']['trans'] = trans_l if trans_l is not None else np.zeros((seq_len, 3))
        mano_data['left']['shape'] = shape_l.flatten()[:10] if shape_l is not None else np.zeros(10)
        mano_data['left']['fitting_err'] = np.zeros(seq_len)

    np.save(os.path.join(output_dir, f"{seq_name}.mano.npy"), mano_data)
    print(f"[Success] Saved {seq_name}.mano.npy")

    # ---------------------------------------------------------
    # 3. Save .object.npy
    # ---------------------------------------------------------
    obj_arti = load_pt_as_numpy(os.path.join(source_root, 'preds/pred.object.radian.pt'))
    obj_rot  = load_pt_as_numpy(os.path.join(source_root, 'preds/pred.object.rot.pt'))
    obj_trans= load_pt_as_numpy(os.path.join(source_root, 'preds/pred.object.cam_t.pt'))

    if obj_rot is not None and obj_trans is not None:
        # Normalize shapes
        obj_rot = normalize_shape_N_D(obj_rot, target_dim=3, name="Object Rot")
        obj_trans = normalize_shape_N_D(obj_trans, target_dim=3, name="Object Trans")
        
        # Unit conversion: Meters -> Millimeters
        obj_trans = obj_trans * 1000.0
        
        # Handle Articulation
        if obj_arti is None: 
            obj_arti = np.zeros((1, 1)) # Default 0
        
        # Force 1D -> 2D before normalization
        obj_arti = normalize_shape_N_D(obj_arti, target_dim=1, name="Object Articulation")
        
        # Sync all to seq_len
        obj_rot = synchronize_sequence_length(obj_rot, seq_len, "Obj Rot")
        obj_trans = synchronize_sequence_length(obj_trans, seq_len, "Obj Trans")
        obj_arti = synchronize_sequence_length(obj_arti, seq_len, "Obj Arti")
        
        # Stack: [Arti(1), Rot(3), Trans(3)] -> (N, 7)
        obj_data = np.hstack([obj_arti, obj_rot, obj_trans])
        
        np.save(os.path.join(output_dir, f"{seq_name}.object.npy"), obj_data)
        print(f"[Success] Saved {seq_name}.object.npy (Shape: {obj_data.shape})")

    # ---------------------------------------------------------
    # 4. Save .smplx.npy
    # ---------------------------------------------------------
    # Initialize full structure with Zeros
    smplx_data = {
        'transl': np.zeros((seq_len, 3), dtype=np.float32),
        'global_orient': np.zeros((seq_len, 3), dtype=np.float32),
        'body_pose': np.zeros((seq_len, 63), dtype=np.float32),
        'jaw_pose': np.zeros((seq_len, 3), dtype=np.float32),
        'leye_pose': np.zeros((seq_len, 3), dtype=np.float32),
        'reye_pose': np.zeros((seq_len, 3), dtype=np.float32),
        'left_hand_pose': np.zeros((seq_len, 45), dtype=np.float32),
        'right_hand_pose': np.zeros((seq_len, 45), dtype=np.float32),
    }

    # Helper to load ref and sync length
    def load_ref_key(ref_dict, key, target_len):
        if key in ref_dict:
            arr = normalize_shape_N_D(ref_dict[key]) # Ensure shapes are sane
            return synchronize_sequence_length(arr, target_len)
        return None

    if ref_smplx_path and os.path.exists(ref_smplx_path):
        try:
            ref_body = np.load(ref_smplx_path, allow_pickle=True).item()
            
            t = load_ref_key(ref_body, 'transl', seq_len)
            o = load_ref_key(ref_body, 'global_orient', seq_len)
            b = load_ref_key(ref_body, 'body_pose', seq_len)
            
            if t is not None: smplx_data['transl'] = t
            if o is not None: smplx_data['global_orient'] = o
            if b is not None: smplx_data['body_pose'] = b
            print(f"  [Info] Stitched Body from {os.path.basename(ref_smplx_path)}")
        except Exception as e:
            print(f"  [Warning] Failed to stitch body: {e}")
    else:
        # Fallback: Attach body to wrist if no reference
        if 'trans' in mano_data['right']:
            smplx_data['transl'] = mano_data['right']['trans']
            smplx_data['global_orient'] = mano_data['right']['rot']

    # Inject Planner Hands
    if pose_l is not None: smplx_data['left_hand_pose'] = pose_l
    if pose_r is not None: smplx_data['right_hand_pose'] = pose_r

    np.save(os.path.join(output_dir, f"{seq_name}.smplx.npy"), smplx_data)
    print(f"[Success] Saved {seq_name}.smplx.npy")

    # ---------------------------------------------------------
    # 5. Save .egocam.dist.npy
    # ---------------------------------------------------------
    # Default Pinhole
    intrinsics = np.array([[1000.0, 0, 500.0], [0, 1000.0, 500.0], [0, 0, 1.0]])

    if ref_egocam_path and os.path.exists(ref_egocam_path):
        try:
            ref_cam = np.load(ref_egocam_path, allow_pickle=True)
            if ref_cam.shape == (): ref_cam = ref_cam.item()
            if 'intrinsics' in ref_cam:
                intrinsics = ref_cam['intrinsics']
                print(f"  [Info] Stitched Intrinsics from {os.path.basename(ref_egocam_path)}")
        except Exception as e:
            print(f"  [Warning] Failed to read reference EgoCam: {e}")

    # Generate Identity Camera Extrinsics (Planner is assumed Camera Space)
    # Be explicit about types to avoid object-array issues
    egocam_data = {
        'R_k_cam_np': np.tile(np.eye(3), (seq_len, 1, 1)).astype(np.float32),
        'T_k_cam_np': np.zeros((seq_len, 3, 1), dtype=np.float32),
        'intrinsics': intrinsics,
        'ego_markers.ref': [[0.0, 0.0, 0.0]] * 5,
        'ego_markers.label': ['M_1', 'M_2', 'M_3', 'M_4', 'M_5'],
        'dist8': np.zeros(8),
        'R0': np.eye(3),
        'T0': np.zeros(3)
    }

    np.save(os.path.join(output_dir, f"{seq_name}.egocam.dist.npy"), egocam_data)
    print(f"[Success] Saved {seq_name}.egocam.dist.npy")
    print("=" * 40)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        src = sys.argv[1]
        ref_smplx = sys.argv[2] if len(sys.argv) > 2 else None
        ref_egocam = sys.argv[3] if len(sys.argv) > 3 else None
        
        target_output = "test_script_output/s05/"
        
        generate_arctic_raw(src, output_dir=target_output, ref_smplx_path=ref_smplx, ref_egocam_path=ref_egocam)
    else:
        print("Usage: python generate_arctic_raw_only.py <source_dir> [ref_smplx.npy] [ref_egocam.npy]")