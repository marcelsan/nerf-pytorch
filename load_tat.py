import glob
import os
import torch
import numpy as np
import imageio 
import torch.nn.functional as F
import cv2

def load_tat_data(basedir, half_res=False):
    Ks = np.load(os.path.join(basedir, 'Ks.npy'))
    Rs = np.load(os.path.join(basedir, 'Rs.npy'))
    Ts = np.load(os.path.join(basedir, 'ts.npy'))
    frames = sorted(glob.glob(os.path.join(basedir, 'images', '*')))

    all_imgs = []
    all_poses = []
    all_frameids = []
    for fname in frames:
        frame_id = int(os.path.splitext(fname)[0].split("_")[-1])
        img = (imageio.imread(fname) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        
        # camera matrix.
        R = Rs[frame_id]
        t = Ts[frame_id]
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = (t.reshape(3, 1)).ravel()

        all_imgs.append(img)
        all_poses.append(pose)
        all_frameids.append(frame_id)
    
    imgs = np.array(all_imgs)
    poses = np.array(all_poses)
    frameids = np.array(all_frameids)

    H, W = imgs[0].shape[:2]
    focal = Ks[0][0, 0]
        
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            # According to the api defined in the link below, the dimension 
            # should be represented as (W, H).
            # https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            # imgs_half_res[i] = cv2.resize(img, (H, W), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, frameids, [H, W, focal]