import os

import collections
import copy
import sys
import time
from random import seed
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from torch import optim
import pandas as pd
import numpy as np
import time
import csv
import dataset
# import evaluation
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from helpers import *
from UNet import UNetModel, update_ema_params

import torch
import torch.nn as nn
# from data_loader import MVTecDRAEMTrainDataset
from torch.utils.data import DataLoader
from torch import optim
# from tensorboard_visualizer import TensorboardVisualizer
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from loss import FocalLoss, SSIM

torch.cuda.empty_cache()

ROOT_DIR = "./"


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(training_dataset, testing_dataset, args, resume):
    training_dataset_loader = dataset.init_dataset_loader(training_dataset, args)
    testing_dataset_loader = dataset.init_dataset_loader(testing_dataset, args)

    in_channels = 1
    if args["dataset"].lower() == "cifar" or args["dataset"].lower() == "leather":
        in_channels = 3

    if args["channels"] != "":
        in_channels = args["channels"]

    model = UNetModel(
        args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], dropout=args[
            "dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
        in_channels=in_channels
    )

    device_ids = args['device_ids']  # 指定可用设备编号

    # model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)

    model_seg = DiscriminativeSubNetwork(in_channels=2, out_channels=1)
    # model_seg = nn.DataParallel(model_seg, device_ids=device_ids)
    model_seg.to(device)
    model_seg.apply(weights_init)

    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": args['lr'], "weight_decay": args['weight_decay'], "betas": (0.9, 0.999)},
        {"params": model_seg.parameters(), "lr": args['lr'], "weight_decay": args['weight_decay'],
         "betas": (0.9, 0.999)}])
    # optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
    # optimiser = optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'], betas=(0.9, 0.999))

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [args['EPOCHS'] * 0.8, args['EPOCHS'] * 0.9], gamma=0.2,
                                               last_epoch=-1)

    loss_l2 = torch.nn.modules.loss.MSELoss()
    loss_ssim = SSIM()
    loss_focal = FocalLoss()

    betas = get_beta_schedule(args['T'], args['beta_schedule'])  # 1000 'linear'

    diffusion = GaussianDiffusionModel(
        args['img_size'], betas, loss_weight=args['loss_weight'],
        loss_type=args['loss-type'], noise=args["noise_fn"], img_channels=in_channels
    )

    # if resume:  # if continuing training from checkpoint
    #
    #     if "unet" in resume:
    #         model.load_state_dict(resume["unet"])
    #     else:
    #         model.load_state_dict(resume["ema"])
    #
    #     ema = UNetModel(
    #         args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'],
    #         dropout=args["dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
    #         in_channels=in_channels
    #     )
    #     ema.load_state_dict(resume["ema"])
    #     start_epoch = resume['n_epoch']
    #
    # else:
    #     start_epoch = 0
    #     ema = copy.deepcopy(model)
    #
    # # ema = nn.DataParallel(ema, device_ids=device_ids)
    # ema.to(device)

    start_epoch = 0
    tqdm_epoch = range(start_epoch, args['EPOCHS'] + 1)

    if resume:
        optimizer.load_state_dict(resume["optimizer_state_dict"])

    del resume

    start_time = time.time()
    losses = []
    l2_losses = []
    ssim_losses = []
    segment_losses = []
    noise_losses = []
    vlb = collections.deque([], maxlen=10)
    iters = range(100 // args['Batch_Size']) if args["dataset"].lower() != "cifar" else range(200)
    d_set_size = len(training_dataset) // args['Batch_Size']


    try:
        os.makedirs(f'./training_outputs/ARGS={args["arg_num"]}_t=100_num={args["ex_num"]}')
    except OSError:
        pass

    training_csv = f'./training_outputs/ARGS={args["arg_num"]}_t=100_num={args["ex_num"]}/training_results.csv'
    with open(training_csv, 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Total VLB', 'Mean Total VLB', 'Prior VLB', 'VB', 'X_0 MSE', 'MSE', 'Elapsed Time', 'Estimated Time Remaining']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


    # dataset loop
    for epoch in tqdm_epoch:
        mean_loss = []
        mean_l2_loss = []
        mean_ssim_loss = []
        mean_segment_loss = []
        mean_noise_loss = []
        print('epoch:', epoch)
        # for i in iters:
        for i in range(d_set_size):
            print('epoch:', epoch, "  ", i, "/", d_set_size)
            data = next(training_dataset_loader)

            x = data["image"].to(device)  # torch.Size([1, 1, 256, 256])
            input = data["input"].to(device)
            # input, healthy_masks, unhealthy_masks = dataset.get_IPM_FPM_image(x, "Train", data['id'])
            box_healthy_mask = data["box_healthy_mask"].to(device)
            box_unhealthy_mask = data["box_unhealthy_mask"].to(device)
            anomaly_mask = data["unhealthy_mask"].to(device)

            # anomaly_mask = torch.where(anomaly_mask > 0, torch.tensor(1).to(device), torch.tensor(0).to(device))
            # anomaly_mask = anomaly_mask.float()

            # save_dir = f'./diffusion-detection-images/ARGS={args["arg_num"]}/'
            #
            # try:
            #     os.makedirs(save_dir)
            # except OSError:
            #     pass

            # ---------------DDPM noise loss----------------------

            # break

            noise_loss, estimates = diffusion.p_loss(model, input, args)
            # noisy, est = estimates[1], estimates[2]

            # --------------------------------------

            # noise = torch.rand_like(augmented_image)
            # t = torch.randint(0, diffusion.num_timesteps, (augmented_image.shape[0],), device=augmented_image.device)
            # x_t = diffusion.sample_q(augmented_image, t, noise)
            # temp = diffusion.sample_p(ema, x_t, t)
            # gray_rec = temp["pred_x_0"]
            t_distance = 200
            gray_rec_seq = diffusion.forward_backward(
                model, input,
                see_whole_sequence="half",
                t_distance=t_distance, denoise_fn=args["noise_fn"]
            )
            gray_rec = gray_rec_seq[-1].to(device)
            x_t = gray_rec_seq[1].to(device)

            mse = (gray_rec - x).square()
            # -------segment_loss----------------------------

            # joined_in = torch.cat((gray_rec, x), dim=1)
            # out_mask = torch.sigmoid(model_seg(joined_in))
            # show_out_mask = (out_mask > 0.5).float()
            # segment_loss = loss_l2(out_mask, box_healthy_mask)

            joined_in = torch.cat((gray_rec*(1-box_unhealthy_mask), x*(1-box_unhealthy_mask)), dim=1)
            out_mask = torch.sigmoid(model_seg(joined_in))
            show_out_mask = (out_mask > 0.5).float()
            segment_loss = loss_l2(out_mask, box_healthy_mask)
            # -----------------------------------

            # # healthy_mask = 1 - anomaly_mask
            #
            #
            healthy_gray_rec = gray_rec * (1 - box_unhealthy_mask)
            healthy_x = x * (1 - box_unhealthy_mask)
            #
            l2_loss = loss_l2(healthy_gray_rec, healthy_x)
            ssim_loss = loss_ssim(healthy_gray_rec, healthy_x)

            # l2_loss = loss_l2(gray_rec, x)
            # -------loss-------------------------

            loss = l2_loss + segment_loss
            # loss = l2_loss + segment_loss + noise_loss
            # loss = l2_loss + ssim_loss + segment_loss
            # loss = l2_loss + ssim_loss + segment_loss + noise_loss
            # loss = l2_loss + ssim_loss + 1000 * segment_loss
            # print(loss)
            # ------------------------------------
            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(model_seg.parameters(), 1)

            optimizer.step()

            # update_ema_params(ema, model)
            # ema = copy.deepcopy(model)

            mean_loss.append(loss.data.cpu())
            mean_l2_loss.append(l2_loss.data.cpu())
            # mean_ssim_loss.append(ssim_loss.data.cpu())
            mean_segment_loss.append(segment_loss.data.cpu())
            # mean_noise_loss.append(noise_loss.data.cpu())

            if epoch % 5 == 0 and i <= 5:
                # row_size = min(8, args['Batch_Size'])
                # training_outputs(
                #         diffusion, augmented_image, est, noisy, epoch, row_size, save_imgs=args['save_imgs'],
                #         save_vids=args['save_vids'], ema=ema, args=args
                #         )
                # -------draw----------------------------
                draw_imgs = True

                if draw_imgs:

                    out = {
                        'image': x[args['Batch_Size']-1:],
                        'input': input[args['Batch_Size']-1:],
                        # 'healthy_masks': box_healthy_mask[args['Batch_Size'] - 1:],
                        # 'unhealthy_masks':box_unhealthy_mask[args['Batch_Size'] - 1:],
                        # 'augmented_image': augmented_image,
                        # 'vertebrae_mask': vertebrae_mask,

                        'x_t': x_t[args['Batch_Size']-1:],
                        'gray_rec': gray_rec[args['Batch_Size']-1:],
                        # 'healthy_gray_rec': healthy_gray_rec[args['Batch_Size']-1:],
                        'mse_heatmap': mse[args['Batch_Size'] - 1:],
                        'out_mask': out_mask[args['Batch_Size']-1:],


                        # "show_out_mask":show_out_mask[args['Batch_Size']-1:],
                        # 'anomaly_mask': anomaly_mask[args['Batch_Size']-1:],
                        'unhealthy_masks':box_unhealthy_mask[args['Batch_Size'] - 1:],
                        "box_healthy_mask":box_healthy_mask[args['Batch_Size'] - 1:],
                        # 'aug': aug,
                        # 'gray_image': gray_image,
                    }

                    row_size = len(out.keys())

                    # plt.close()  # 关闭之前的Figure

                    # create a figure with one row and row_size columns
                    width = 2
                    fig, axes = plt.subplots(nrows=1, ncols=row_size, figsize=(width * row_size, width))
                    # fig, axes = plt.subplots(nrows=1, ncols=row_size)
                    # 调整子图之间的距离和位置
                    fig.subplots_adjust(wspace=0)
                    # loop through the tensors and plot them on each subplot
                    j = 0
                    for name, tensor in out.items():
                        # print(name)
                        # convert the tensor to a numpy array
                        array = tensor[0][0].cpu().detach().numpy()

                        if name == 'mse_heatmap' or name == 'out_mask':
                            # 缩放mse到[0,255]范围内
                            scaled_mse = array * 255
                            # 可视化mse
                            sns.heatmap(data=scaled_mse, ax=axes[j], cmap='hot', cbar=False)
                        else:
                            axes[j].imshow(array, cmap='gray')
                        # optionally add a title for each subplot
                        axes[j].set_title(name)
                        axes[j].axis('off')
                        j = j + 1

                    # plt.suptitle( new["id"][0] + 'patient', y=0)
                    plt.suptitle(str(data["id"][0]))
                    plt.rcParams['figure.dpi'] = 150

                    save_dir = f'./training_outputs/ARGS={args["arg_num"]}_t={t_distance}_num={args["ex_num"]}/epoch={epoch}'
                    try:
                        os.makedirs(save_dir)
                    except OSError:
                        pass

                    plt.savefig(save_dir + "/iter_" + str(i) + '.png')

                    print(save_dir + "/iter_" + str(i) + '.png')
                    print("training output has been saved!")
                    # plt.show()

                    plt.clf()

        losses.append(np.mean(mean_loss))
        l2_losses.append(np.mean(mean_l2_loss))
        ssim_losses.append(np.mean(mean_ssim_loss))
        segment_losses.append(np.mean(mean_segment_loss))
        noise_losses.append(np.mean(mean_noise_loss))

        print("loss:", losses[-1])
        print("l2_losses:", l2_losses[-1])
        # print("ssim_losses:", ssim_losses[-1])
        print("segment_losses:", segment_losses[-1])
        # print("noise_losses:", noise_losses[-1])

        if epoch % 10 == 0:
            time_taken = time.time() - start_time
            remaining_epochs = args['EPOCHS'] - epoch
            time_per_epoch = time_taken / (epoch + 1 - start_epoch)
            hours = remaining_epochs * time_per_epoch / 3600
            mins = (hours % 1) * 60
            hours = int(hours)

            vlb_terms = diffusion.calc_total_vlb(x, model, args)
            vlb.append(vlb_terms["total_vlb"].mean(dim=-1).cpu().item())
            print(
                f"epoch: {epoch}, most recent total VLB: {vlb[-1]} mean total VLB:"
                f" {np.mean(vlb):.4f}, "
                f"prior vlb: {vlb_terms['prior_vlb'].mean(dim=-1).cpu().item():.2f}, vb: "
                f"{torch.mean(vlb_terms['vb'], dim=list(range(2))).cpu().item():.2f}, x_0_mse: "
                f"{torch.mean(vlb_terms['x_0_mse'], dim=list(range(2))).cpu().item():.2f}, mse: "
                f"{torch.mean(vlb_terms['mse'], dim=list(range(2))).cpu().item():.2f}"
                f" time elapsed {int(time_taken / 3600)}:{((time_taken / 3600) % 1) * 60:02.0f}, "
                f"est time remaining: {hours}:{mins:02.0f}\r"
            )
            with open(training_csv, 'a', newline='') as csvfile:
                fieldnames = ['Epoch', 'Total VLB', 'Mean Total VLB', 'Prior VLB', 'VB', 'X_0 MSE', 'MSE',
                              'Elapsed Time', 'Estimated Time Remaining']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                elapsed_time = int(time_taken / 3600), ((time_taken / 3600) % 1) * 60
                estimated_time_remaining = hours, mins

                writer.writerow({
                    'Epoch': epoch,
                    'Total VLB': vlb[-1],
                    'Mean Total VLB': np.mean(vlb),
                    'Prior VLB': vlb_terms['prior_vlb'].mean(dim=-1).cpu().item(),
                    'VB': torch.mean(vlb_terms['vb'], dim=list(range(2))).cpu().item(),
                    'X_0 MSE': torch.mean(vlb_terms['x_0_mse'], dim=list(range(2))).cpu().item(),
                    'MSE': torch.mean(vlb_terms['mse'], dim=list(range(2))).cpu().item(),
                    'Elapsed Time': f"{elapsed_time[0]}:{elapsed_time[1]:02.0f}",
                    'Estimated Time Remaining': f"{estimated_time_remaining[0]}:{estimated_time_remaining[1]:02.0f}"
                })

        # if epoch % 100 == 0 and epoch > 0:
        if epoch % 10 == 0 and epoch != args['EPOCHS']:
            scheduler.step()
            save(unet=model, args=args, optimiser=optimizer, final=False, epoch=epoch, loss=losses[-1].item())
            seg_save(model=model_seg, args=args, optimiser=optimizer, final=False, epoch=epoch, loss=losses[-1].item())

    scheduler.step()
    save(unet=model, args=args, optimiser=optimizer, final=True)
    seg_save(model=model_seg, args=args, optimiser=optimizer, final=True, epoch=epoch)

    # evaluation.testing(testing_dataset_loader, diffusion, ema=ema, args=args, model=model)

    # torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name + ".pckl"))
    # torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, run_name + "_seg.pckl"))


def save(final, unet, optimiser, args, loss=0, epoch=0):
    if final:
        torch.save(
            {
                'n_epoch': args["EPOCHS"],
                'model_state_dict':     unet.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                # "ema": ema.state_dict(),
                "args": args
                # 'loss': LOSS,
            }, f'{ROOT_DIR}model/params-ARGS={args["arg_num"]}_num={args["ex_num"]}/params-final.pt'
        )
    else:
        torch.save(
            {
                'n_epoch': epoch,
                'model_state_dict':     unet.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                "args": args,
                # "ema": ema.state_dict(),
                'loss': loss,
            }, f'{ROOT_DIR}model/params-ARGS={args["arg_num"]}_num={args["ex_num"]}/checkpoint/diff/diff_epoch={epoch}.pt'
        )


def seg_save(final, model, optimiser, args, loss=0, epoch=0):

    if final:
        torch.save(
            {
                'n_epoch': args["EPOCHS"],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                "args": args
            }, f'{ROOT_DIR}model/params-ARGS={args["arg_num"]}_num={args["ex_num"]}/seg_params-final.pt'
        )
    else:
        torch.save(
            {
                'n_epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                "args": args,
                'loss': loss,
            }, f'{ROOT_DIR}model/params-ARGS={args["arg_num"]}_num={args["ex_num"]}/checkpoint/seg/seg_epoch={epoch}.pt'
        )


def main():
    # make directories
    for i in ['./model/', "./training_outputs/"]:
        try:
            os.makedirs(i)
        except OSError:
            pass

    # read file from argument
    if len(sys.argv[1:]) > 0:
        files = sys.argv[1:]
    else:
        raise ValueError("Missing file argument")

    # resume from final or resume from most recent checkpoint -> ran from specific slurm script?
    resume = 0
    if files[0] == "RESUME_RECENT":
        resume = 1
        files = files[1:]
        if len(files) == 0:
            raise ValueError("Missing file argument")
    elif files[0] == "RESUME_FINAL":
        resume = 2
        files = files[1:]
        if len(files) == 0:
            raise ValueError("Missing file argument")

    # allow different arg inputs ie 25 or args15 which are converted into argsNUM.json
    file = files[0]
    if file.isnumeric():
        file = f"args{file}.json"
    elif file[:4] == "args" and file[-5:] == ".json":
        pass
    elif file[:4] == "args":
        file = f"args{file[4:]}.json"
    else:
        raise ValueError("File Argument is not a json file")

    # load the json args       file = 'args28.json'
    with open(f'{ROOT_DIR}configs/{file}', 'r') as f:
        args = json.load(f)
    args['arg_num'] = file[4:-5]
    args = defaultdict_from_json(args)

    # make arg specific directories
    for i in [f'./model/params-ARGS={args["arg_num"]}_num={args["ex_num"]}',
              f'./model/params-ARGS={args["arg_num"]}_num={args["ex_num"]}/checkpoint/seg',
              f'./model/params-ARGS={args["arg_num"]}_num={args["ex_num"]}/checkpoint/diff',
              ]:
        try:
            os.makedirs(i)
        except OSError:
            pass

    print(file, args)

    if args["channels"] != "":
        in_channels = args["channels"]

    # load my vertebrae dataset
    training_dataset, testing_dataset = dataset.diff_seg_datasets(ROOT_DIR, args)

    # if resuming, loaded model is attached to the dictionary
    loaded_model = {}
    if resume:
        if resume == 1:  # 加载最近一个checkpoint
            checkpoints = os.listdir(f'./model/params-ARGS={args["arg_num"]}_num={args["ex_num"]}/checkpoint')
            checkpoints.sort(reverse=True)
            for i in checkpoints:
                try:
                    file_dir = f"./model/params-ARGS={args['arg_num']}/checkpoint/{i}"
                    loaded_model = torch.load(file_dir, map_location=device)
                    break
                except RuntimeError:
                    continue

        else:  # 加载params-final.pt
            file_dir = f'./model/params-ARGS={args["arg_num"]}_num={args["ex_num"]}/params-final.pt'
            loaded_model = torch.load(file_dir, map_location=device)

    # load, pass args
    train(training_dataset, testing_dataset, args, loaded_model)

    # remove checkpoints after final_param is saved (due to storage requirements)
    # for file_remove in os.listdir(f'./model/params-ARGS={args["arg_num"]}_num={args["ex_num"]}/checkpoint'):
    #     os.remove(os.path.join(f'./model/params-ARGS={args["arg_num"]}_num={args["ex_num"]}/checkpoint', file_remove))
    #
    # os.removedirs(f'./model/params-ARGS={args["arg_num"]}_num={args["ex_num"]}/checkpoint')


if __name__ == '__main__':
    # import os
    #
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # torch.backends.cudnn.benchmark = True  # 将根据输入的大小确定卷积操作最优算法的使用设置为真
    # # torch.backends.cuda.max_split_size = 1024  # 设置PyTorch自动分配的最大内存大小，单位为MB
    # torch.backends.cuda.max_split_size_mb = 1024

    seed(1)

    main()
