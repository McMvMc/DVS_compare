import os, sys
from datetime import datetime
import numpy as np
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import cv2
from kornia import create_meshgrid

import pytorch3d as p3d
from pytorch3d.structures import Pointclouds

from render_utils import *
from run_nerf_helpers import *
from load_llff import *

# from evaluation import calculate_ssim, calculate_psnr
# import skimage.measure
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
loss_fn_alex = lpips.LPIPS(net='alex').cuda()  # best forward scores

import matplotlib.pyplot as plt


USE_SINGLE_GPU = False
WORLD_SIZE = 4
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(1)
DEBUG = False


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=300000,
                        help='exponential learning rate decay')
    parser.add_argument("--chunk", type=int, default=1024*64,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--random_seed", type=int, default=1,
                        help='fix random seed for repeatability')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--use_viewdirsDyn", action='store_true',
                        help='use full 5D input instead of 3D for D-NeRF')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff')

    # llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=2000,
                        help='frequency of console printout and metric logging')
    parser.add_argument("--i_img",     type=int, default=2000,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=5000,
                        help='frequency of render_poses video saving')
    parser.add_argument("--N_iters", type=int, default=1000000,
                        help='number of training iterations')

    # Dynamic NeRF lambdas
    parser.add_argument("--dynamic_loss_lambda", type=float, default=1.,
                        help='lambda of dynamic loss')
    parser.add_argument("--static_loss_lambda", type=float, default=1.,
                        help='lambda of static loss')
    parser.add_argument("--full_loss_lambda", type=float, default=3.,
                        help='lambda of full loss')
    parser.add_argument("--depth_loss_lambda", type=float, default=0.04,
                        help='lambda of depth loss')
    parser.add_argument("--order_loss_lambda", type=float, default=0.1,
                        help='lambda of order loss')
    parser.add_argument("--flow_loss_lambda", type=float, default=0.02,
                        help='lambda of optical flow loss')
    parser.add_argument("--slow_loss_lambda", type=float, default=0.1,
                        help='lambda of sf slow regularization')
    parser.add_argument("--smooth_loss_lambda", type=float, default=0.1,
                        help='lambda of sf smooth regularization')
    parser.add_argument("--consistency_loss_lambda", type=float, default=0.1,
                        help='lambda of sf cycle consistency regularization')
    parser.add_argument("--mask_loss_lambda", type=float, default=0.1,
                        help='lambda of the mask loss')
    parser.add_argument("--sparse_loss_lambda", type=float, default=0.1,
                        help='lambda of sparse loss')
    parser.add_argument("--DyNeRF_blending", action='store_true',
                        help='use Dynamic NeRF to predict blending weight')
    parser.add_argument("--pretrain", action='store_true',
                        help='Pretrain the StaticneRF')
    parser.add_argument("--ft_path_S", type=str, default=None,
                        help='specific weights npy file to reload for StaticNeRF')

    # For rendering teasers
    parser.add_argument("--frame2dolly", type=int, default=-1,
                        help='choose frame to perform dolly zoom')
    parser.add_argument("--x_trans_multiplier", type=float, default=1.,
                        help='x_trans_multiplier')
    parser.add_argument("--y_trans_multiplier", type=float, default=0.33,
                        help='y_trans_multiplier')
    parser.add_argument("--z_trans_multiplier", type=float, default=5.,
                        help='z_trans_multiplier')
    parser.add_argument("--num_novelviews", type=int, default=60,
                        help='num_novelviews')
    parser.add_argument("--focal_decrease", type=float, default=200,
                        help='focal_decrease')
    parser.add_argument("--final_height", default=504)
    parser.add_argument("--final_width", default=896)
    parser.add_argument("--seq_num")
    parser.add_argument("--setup_num")
    parser.add_argument("--start_frame", type=int)
    parser.add_argument("--end_frame", type=int)
    parser.add_argument("--target_idx", type=int)
    return parser


def render_freeview(all_poses, args, n_frame_each_edge, rank):
    val_poses = []
    for val_i_tmp in range(n_frame_each_edge*3):
        # images = torch.Tensor(all_images[args.serial_list[0]]).cuda()  # .to(device)
        # depths = torch.Tensor(all_depths[args.serial_list[0]]).cuda()  # .to(device)
        # poses = torch.Tensor(all_poses[args.serial_list[0]]).cuda()
        if (val_i_tmp % (n_frame_each_edge * 3)) < n_frame_each_edge:
            s_i = 0
            e_i = 2
        elif (val_i_tmp % (n_frame_each_edge * 3)) < n_frame_each_edge * 2:
            s_i = 2
            e_i = 1
        else:
            s_i = 1
            e_i = 0

        s_pose = torch.Tensor(all_poses[args.serial_list[s_i]]).to(rank)[0]
        e_pose = torch.Tensor(all_poses[args.serial_list[e_i]]).to(rank)[0]

        rot_1 = p3d.transforms.matrix_to_euler_angles(s_pose[:, :3], 'XYZ')
        trans_1 = s_pose[:, 3]
        rot_2 = p3d.transforms.matrix_to_euler_angles(e_pose[:, :3], 'XYZ')
        trans_2 = e_pose[:, 3]
        weight_1 = 1. - (val_i_tmp % n_frame_each_edge) / n_frame_each_edge
        # weight_1 = 0.
        val_cam_pose = torch.zeros((3,4)).to(rank)
        val_cam_pose[:, :3] = p3d.transforms.euler_angles_to_matrix(weight_1 * rot_1 + (1. - weight_1) * rot_2,
                                                                    'XYZ')
        val_cam_pose[:, 3] = weight_1 * trans_1 + (1. - weight_1) * trans_2
        val_poses.append(val_cam_pose)

    return val_poses


def eval():

    parser = config_parser()
    args = parser.parse_args()
    args.rank=None

    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)

    # Load data
    if args.dataset_type == 'llff':
        frame2dolly = args.frame2dolly
        images, invdepths, masks, poses, bds, \
        render_poses, render_focals, grids = load_llff_data(args, args.datadir,
                                                            args.factor,
                                                            frame2dolly=frame2dolly,
                                                            recenter=True, bd_factor=.9,
                                                            spherify=args.spherify)

        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        num_img = float(poses.shape[0])
        assert len(poses) == len(images)
        print('Loaded llff', images.shape,
            render_poses.shape, hwf, args.datadir)

        # Use all views to train
        i_train = np.array([i for i in np.arange(int(images.shape[0]))])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            raise NotImplementedError
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)
    else:
        args.datadir += f'/seq_{args.seq_num}/'
        args.serial_list = ['016334443347', '025289251247', '029090143247']
        args.gt_serial = '029090143247'
        all_images, all_depths, all_masks, all_poses, all_bds, all_render_poses, \
        all_ref_c2w, all_grids, all_dist_masks = load_DFVS_data(args.datadir,
                                                            args.start_frame, args.end_frame,
                                                            args,
                                                            target_idx=args.target_idx,
                                                            final_height=args.final_height)
        n_frames = int(all_images[args.gt_serial].shape[0])
        i_val = np.array(np.arange(n_frames)) #i_test
        i_train = np.array(np.arange(n_frames))
        i_test = np.array(np.arange(n_frames))

        near = 0.
        far = 1.

    # Create log dir and copy the config file
    basedir = args.basedir
    args.expname = args.expname + '_F%02d-%02d' % (args.start_frame, args.end_frame) \
                   + f'_{datetime.now().strftime("%m-%d_%H-%M")}'
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
        'num_img': n_frames,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)


    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    save_dir = os.path.join(args.datadir, f'dynamic_nerf_{args.start_frame}to{args.end_frame}'
                                          f'_{args.target_idx}')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # poses = all_poses[args.serial_list[0]]
    poses = all_poses[args.gt_serial]
    poses = torch.Tensor(poses).to(device)
    hwf = poses[0, :3, -1]
    H, W, focal = hwf
    H = H.type(torch.int).item()
    W = W.type(torch.int).item()

    # pose_list = render_freeview(all_poses, args, n_frame_each_edge, rank)
    # with torch.no_grad():
    #     for render_i, pose in enumerate(pose_list):
    #         ret = render(t, False, H, W, focal,
    #                      chunk=1024 * 16,
    #                      c2w=pose,
    #                      **render_kwargs_test)
    #         if rank == 0:
    #             if not os.path.exists(save_dir):
    #                 os.mkdir(save_dir)
    #             save_fn = os.path.join(save_dir, f'nv_{render_i:05d}.png')
    #             free_view_img = ret['rgb_map_full'].detach().permute(2, 0, 1)
    #             save_image(free_view_img, save_fn)
    #             time.sleep(3)  # sleep 3 sec to lower temperature

    with torch.no_grad():
        images = all_images[args.gt_serial]
        psnr_list = []
        ssim_list = []
        perceptual_loss_list = []
        time_list = []
        for t in range(n_frames):
            print(f"synthesizing frame {t}")
            gt_img = images[t]
            img_idx_embed = t / n_frames * 2. - 1.0
            pose = poses[t,:3,:4]
            t0 = time.time()
            ret = render(img_idx_embed, False, H, W, focal,
                         chunk=1024 * 16,
                         c2w=pose,
                         **render_kwargs_test)
            cur_time = time.time() - t0
            save_fn = os.path.join(save_dir, f'nv_{t:05d}.png')
            free_view_img = ret['rgb_map_full'].detach().permute(2, 0, 1)
            free_view_img_np = free_view_img.cpu().permute(1,2,0).numpy()

            ssim_score = ssim(gt_img, free_view_img_np, win_size=11, multichannel=True, data_range=1.)
            psnr_score = psnr(gt_img, free_view_img_np, data_range=1.)
            perceptual_loss = loss_fn_alex(torch.tensor(gt_img).permute(2,0,1)[None].cuda(),
                                           torch.tensor(free_view_img_np).permute(2,0,1)[None].cuda()).item()
            ssim_list.append(ssim_score)
            psnr_list.append(psnr_score)
            perceptual_loss_list.append(perceptual_loss)
            time_list.append(cur_time)

            print(f"cur ssim: {ssim_score}, psnr: {psnr_score},"
                  f"LPIPS: {perceptual_loss}, time: {cur_time}")
            save_image(free_view_img, save_fn)

            gt_save_fn = os.path.join(save_dir, f'nv_{t:05d}_gt.png')
            save_image(torch.tensor(gt_img).permute(2,0,1), gt_save_fn)

            free_view_img = ret['rgb_map_d'].detach().permute(2, 0, 1)
            save_fn = os.path.join(save_dir, f'dy_{t:05d}.png')
            save_image(free_view_img, save_fn)

            time.sleep(3)  # sleep 3 sec to lower temperature
        print(f"\navg. ssim: {np.mean(ssim_list)}, psnr: {np.mean(psnr_list)},"
              f"LPIPS: {np.mean(perceptual_loss_list)}, time: {np.mean(time_list)}")

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # if USE_SINGLE_GPU:
    eval()
    # else:
    #     # train_DDP(args)
    #     mp.spawn(train_DDP, nprocs=WORLD_SIZE, join=True)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"