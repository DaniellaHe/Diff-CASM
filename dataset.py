import glob
from perlin import rand_perlin_2d_np
import itertools
import json
import os
import random
from random import randint

import cv2
import imgaug.augmenters as iaa
import nibabel as nib
import numpy as np
import torch
from matplotlib import animation
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def make_pngs_anogan():
    dir = {
        "Train": "./DATASETS/Train", "Test": "./DATASETS/Test",
        "Anomalous": "./DATASETS/CancerousDataset/EdinburghDataset/Anomalous-T1"
    }
    slices = {
        "17904": range(165, 205), "18428": range(177, 213), "18582": range(160, 190), "18638": range(160, 212),
        "18675": range(140, 200), "18716": range(135, 190), "18756": range(150, 205), "18863": range(130, 190),
        "18886": range(120, 180), "18975": range(170, 194), "19015": range(158, 195), "19085": range(155, 195),
        "19275": range(184, 213), "19277": range(158, 209), "19357": range(158, 210), "19398": range(164, 200),
        "19423": range(142, 200), "19567": range(160, 200), "19628": range(147, 210), "19691": range(155, 200),
        "19723": range(140, 170), "19849": range(150, 180)
    }
    center_crop = 235
    import os
    try:
        os.makedirs("./DATASETS/AnoGAN")
    except OSError:
        pass
    # for d_set in ["Train", "Test"]:
    #     try:
    #         os.makedirs(f"./DATASETS/AnoGAN/{d_set}")
    #     except OSError:
    #         pass
    #
    #     files = os.listdir(dir[d_set])
    #
    #     for volume_name in files:
    #         try:
    #             volume = np.load(f"{dir[d_set]}/{volume_name}/{volume_name}.npy")
    #         except (FileNotFoundError, NotADirectoryError) as e:
    #             continue
    #         for slice_idx in range(40, 120):
    #             image = volume[:, slice_idx:slice_idx + 1, :].reshape(256, 192).astype(np.float32)
    #             image = (image * 255).astype(np.int32)
    #             empty_image = np.zeros((256, center_crop))
    #             empty_image[:, 21:213] = image
    #             image = empty_image
    #             center = (image.shape[0] / 2, image.shape[1] / 2)
    #
    #             x = center[1] - center_crop / 2
    #             y = center[0] - center_crop / 2
    #             image = image[int(y):int(y + center_crop), int(x):int(x + center_crop)]
    #             image = cv2.resize(image, (64, 64))
    #             cv2.imwrite(f"./DATASETS/AnoGAN/{d_set}/{volume_name}-slice={slice_idx}.png", image)

    try:
        os.makedirs(f"./DATASETS/AnoGAN/Anomalous")
    except OSError:
        pass
    try:
        os.makedirs(f"./DATASETS/AnoGAN/Anomalous-mask")
    except OSError:
        pass
    files = os.listdir(f"{dir['Anomalous']}/raw_cleaned")
    center_crop = (175, 240)
    for volume_name in files:
        try:
            volume = np.load(f"{dir['Anomalous']}/raw_cleaned/{volume_name}")
            volume_mask = np.load(f"{dir['Anomalous']}/mask_cleaned/{volume_name}")
        except (FileNotFoundError, NotADirectoryError) as e:
            continue
        temp_range = slices[volume_name[:-4]]
        for slice_idx in np.linspace(temp_range.start + 5, temp_range.stop - 5, 4).astype(np.uint16):
            image = volume[slice_idx, ...].reshape(volume.shape[1], volume.shape[2]).astype(np.float32)
            image = (image * 255).astype(np.int32)
            empty_image = np.zeros((max(volume.shape[1], center_crop[0]), max(volume.shape[2], center_crop[1])))

            empty_image[9:165, :] = image
            image = empty_image
            center = (image.shape[0] / 2, image.shape[1] / 2)

            x = center[1] - center_crop[1] / 2
            y = center[0] - center_crop[0] / 2
            image = image[int(y):int(y + center_crop[0]), int(x):int(x + center_crop[1])]
            image = cv2.resize(image, (64, 64))
            cv2.imwrite(f"./DATASETS/AnoGAN/Anomalous/{volume_name}-slice={slice_idx}.png", image)

            image = volume_mask[slice_idx, ...].reshape(volume.shape[1], volume.shape[2]).astype(np.float32)
            image = (image * 255).astype(np.int32)
            empty_image = np.zeros((max(volume.shape[1], center_crop[0]), max(volume.shape[2], center_crop[1])))

            empty_image[9:165, :] = image
            image = empty_image
            center = (image.shape[0] / 2, image.shape[1] / 2)

            x = center[1] - center_crop[1] / 2
            y = center[0] - center_crop[0] / 2
            image = image[int(y):int(y + center_crop[0]), int(x):int(x + center_crop[1])]
            image = cv2.resize(image, (64, 64))
            cv2.imwrite(f"./DATASETS/AnoGAN/Anomalous-mask/{volume_name}-slice={slice_idx}.png", image)

def scale_array(array, new_min=0, new_max=255):
    """
    将数组的值缩放到指定范围内。

    参数:
    - array: 输入的NumPy数组
    - new_min: 新范围的最小值
    - new_max: 新范围的最大值

    返回:
    缩放后的数组
    """
    # 找到数组的最小值和最大值
    min_val = np.min(array)
    max_val = np.max(array)

    # 线性变换，将数组的值缩放到新范围内
    scaled_array = ((array - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min

    # 将浮点数数组转换为整数数组
    scaled_array = scaled_array.astype(np.uint8)

    return scaled_array
def get_freq_image(slice):
    # slice = slice[0]
    val = random.randint(2, 10)

    # 计算输入切片的2D傅里叶变换
    fft_image = np.fft.fftshift(np.fft.fft2(slice))

    # 创建二进制掩码或反向二进制掩码
    mask = np.zeros_like(fft_image)
    rows, cols = fft_image.shape
    center_row, center_col = rows // 2, cols // 2
    mask_range = range(center_row - val, center_row + val)

    # 随机选择0或1
    choice = random.randint(0, 1)

    # if choice == 0:
    #     mask[mask_range, center_col - val: center_col + val] = 1
    # else:
    mask[mask_range, center_col - val: center_col + val] = 1
    mask = 1 - mask  # 反向二进制掩码

    # # 使用 matplotlib 进行可视化
    # plt.figure(figsize=(5, 5))
    #
    # # 可视化第一个数组
    # plt.subplot(1, 2, 1)
    # plt.imshow(mask.astype(float), cmap='gray')
    # plt.title('Array 1')
    #
    # plt.show()

    # 将掩码应用于傅里叶变换
    masked_fft = np.multiply(fft_image, mask)

    # 计算逆傅里叶变换以获取重建图像
    reconstructed_image = np.real(np.fft.ifft2(np.fft.ifftshift(masked_fft)))

    return reconstructed_image
def visualize_IPM_FPM_image(img, fft_image, healthy_mask, unhealthy_mask, id):
    scale_fft_image = scale_array(fft_image)
    scale_img = scale_array(img)
    rate = 0.5
    perlin_scale = 6
    min_perlin_scale = 0

    perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
    perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

    perlin_noise = rand_perlin_2d_np((img.shape[0], img.shape[1]), (perlin_scalex, perlin_scaley))
    rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
    perlin_noise = rot(image=perlin_noise)
    threshold = 0.5
    perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
    # perlin_thr = np.expand_dims(perlin_thr, axis=2)
    perlin_thr = torch.from_numpy(perlin_thr)
    msk = perlin_thr.float()
    msk = msk * healthy_mask
    msk = msk.numpy()
    beta = torch.rand(1).numpy()[0] * 0.2 + 0.6

    scale_masked_image = scale_fft_image * msk * beta + scale_img * msk * (1-beta) + scale_img * (1 - msk)
    masked_image = scale_masked_image * (1 - unhealthy_mask)
    masked_image = masked_image.astype(np.float32)

    vshow = True
    # vshow = False
    if vshow:
        # 使用 matplotlib 进行可视化
        plt.figure(figsize=(5, 5))
        folder_path = f'./data/IPM_FPM_image/{id}/'
        os.makedirs(folder_path, exist_ok=True)

        pic = {
            'image': scale_img,
            'fft_image': scale_fft_image,
            'perlin_thr': perlin_thr.numpy(),
            'healthy_mask01': healthy_mask,
            'mask01': msk,
            'Imask01': 1-msk,
            'unhealthy_mask01': unhealthy_mask,
            'Iunhealthy_mask01': 1-unhealthy_mask,
            'masked_image': masked_image,
            'maskunhealthy_image': (1-unhealthy_mask)*scale_img,
        }

        for title, img in pic.items():
            plt.imshow(img, cmap='gray')  # 假设这里的图像是灰度图
            plt.axis('off')
            img = img.astype(np.uint8)

            # 裁剪图像，使其成为正方形
            height, width = img.shape
            min_dim = min(height, width)
            cropped_img = img[:min_dim, :min_dim]
            if 'mask01' in title:
                cropped_img = (cropped_img * 255)
            # 使用PIL库将NumPy数组转换为图像对象，并保存
            image_to_save = Image.fromarray(cropped_img)
            image_to_save.save(folder_path + f'{title}.png')

            plt.close()  # 关闭当前图形以便绘制下一个图像

        data = {
            'image': scale_img,
            'fft_image': scale_fft_image,
            'perlin_thr': perlin_thr.numpy(),
            'healthy_mask': healthy_mask,
            'msk': msk,
            'unhealthy_mask': unhealthy_mask,
            'masked_image': masked_image,
        }
        height = 2
        width = height * len(data)
        plt.figure(figsize=(width, height))  # 设置整个图像的大小
        for i, (title, img) in enumerate(data.items()):
            plt.subplot(1, len(data), i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(title)
            plt.axis('off')

        folder_path = f'./data/IPM_FPM_image/show/'
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(folder_path + f'{id}.png')
        # plt.show()

    return masked_image, msk
def get_IPM_FPM_image(img, Train_or_Test, id):
    fft_image = get_freq_image(img)
    # 初始化当前图像的健康和不健康掩码
    healthy_mask = np.zeros_like(img)
    unhealthy_mask = np.zeros_like(img)

    # 从JSON文件加载裁剪的边界框数据  /home/ub/妗岄潰/code/VSDiff/
    json_file_path = f"../data/VerTumor600/cropped_vertebrae/cropped_bbox.json"
    with open(json_file_path, "r") as json_file:
        loaded_data = json.load(json_file)

    # 查找当前图像的边界框
    bbox = None
    for b in loaded_data:
        if b['id'] == int(id.split('_')[-1]):
            bbox = b['cropped_bbox']
            image_size = b['size']
            if not(image_size[0] == img.shape[0] and image_size[1] == img.shape[1]):
                raise ValueError(f"size != (img.shape[0], img.shape[1])")
            break

    # 检查是否在加载的数据中找到了当前id
    if bbox is None:
        raise ValueError(f"未找到id {id} 的边界框数据")

    # 根据每个边界框中的标签应用掩码
    for (label, l, cropped_bbox) in bbox:
        (x_min, y_min, x_max, y_max) = cropped_bbox
        if label == 1:
            # 不健康掩码
            unhealthy_mask[y_min:y_max, x_min:x_max] = 1
        else:
            healthy_mask[y_min:y_max, x_min:x_max] = 1
            # # 随机选择是否应用健康掩码
            # choice = random.choice([True, False])
            # if choice:
            #     healthy_mask[y_min:y_max, x_min:x_max] = 1

    masked_image, msk = visualize_IPM_FPM_image(img, fft_image, healthy_mask, unhealthy_mask, id)

    # # 将掩码应用于FFT图像并进行逆FFT
    # masked_image = fft_image * healthy_mask + img * (1 - healthy_mask)
    # masked_image = masked_image * (1 - unhealthy_mask)
    # masked_image = masked_image.astype(np.float32)

    return masked_image, msk, unhealthy_mask

def main_get_IPM_FPM_image(images, Train_or_Test, ids):
    # 初始化健康和不健康掩码
    healthy_masks = []
    unhealthy_masks = []
    masked_images = []

    for img, current_id in zip(images, ids):
        # 对每个输入图像进行FFT
        # img_np = img.cpu().numpy()
        img_np = img
        fft_image = np.fft.fftshift(np.fft.fft2(img_np))

        # 初始化当前图像的健康和不健康掩码
        healthy_mask = np.zeros_like(img_np)
        unhealthy_mask = np.zeros_like(img_np)

        # 从JSON文件加载裁剪的边界框数据
        json_file_path = f"./data/cropped_vertebrae/{Train_or_Test}_cropped_bbox.json"
        with open(json_file_path, "r") as json_file:
            loaded_data = json.load(json_file)

        # 查找当前图像的边界框
        bbox = None
        for b in loaded_data:
            if b['id'] == current_id:
                bbox = b['cropped_bbox']
                break

        # 检查是否在加载的数据中找到了当前id
        if bbox is None:
            raise ValueError(f"未找到id {current_id} 的边界框数据")

        # 根据每个边界框中的标签应用掩码
        for (label, cropped_bbox, size) in bbox:
            if size != (img_np.shape[0], img_np.shape[1]):
                raise ValueError(f"size != (img_np.shape[0], img_np.shape[1])")
            (x_min, y_min, x_max, y_max) = cropped_bbox
            if label == 1:
                # 不健康掩码
                unhealthy_mask[y_min:y_max, x_min:x_max] = 1
                # unhealthy_mask[x_min:x_max, y_min:y_max] = 1
            else:
                # 随机选择是否应用健康掩码
                choice = random.choice([True, False])
                if choice:
                    healthy_mask[y_min:y_max, x_min:x_max] = 1

        # 将掩码应用于FFT图像并进行逆FFT
        masked_image = np.real(np.fft.ifft2(np.fft.ifftshift(fft_image * (1 - healthy_mask))))
        # print(masked_image.dtype)
        masked_image = masked_image * (1 - unhealthy_mask)
        masked_image = masked_image.astype(np.float32)
        # print(masked_image.dtype)

        # 将结果添加到列表中
        healthy_masks.append(healthy_mask)
        unhealthy_masks.append(unhealthy_mask)
        masked_images.append(torch.tensor(masked_image))

    # 将结果转换回 PyTorch 张量
    healthy_masks = torch.tensor(healthy_masks)
    unhealthy_masks = torch.tensor(unhealthy_masks)
    masked_images = torch.stack(masked_images)

    return masked_images, healthy_masks, unhealthy_masks

def MY_get_IPM_FPM_image(image, Train_or_Test, id):

    # 对输入图像进行FFT
    image = image.cpu().numpy()
    fft_image = np.fft.fftshift(np.fft.fft2(image))

    # 初始化健康和不健康掩码
    healthy_mask = np.zeros_like(fft_image)
    unhealthy_mask = np.zeros_like(fft_image)

    # 从JSON文件加载裁剪的边界框数据
    json_file_path = f"./data/cropped_vertebrae/{Train_or_Test}_cropped_bbox.json"
    with open(json_file_path, "r") as json_file:
        loaded_data = json.load(json_file)

    # 查找指定id的边界框
    bbox = None
    for b in loaded_data:
        if b['id'] == id:
            bbox = b['cropped_bbox']
            break

    # 检查是否在加载的数据中找到了指定的id
    if bbox is None:
        raise ValueError(f"未找到id {id} 的边界框数据")

    # 根据每个边界框中的标签应用掩码
    for (label, cropped_bbox) in bbox:
        (x_min, y_min, x_max, y_max) = cropped_bbox
        if label == 1:
            # 不健康掩码
            unhealthy_mask[y_min:y_max, x_min:x_max] = 1
        else:
            # 随机选择是否应用健康掩码
            choice = random.choice([True, False])
            if choice:
                healthy_mask[y_min:y_max, x_min:x_max] = 1

    # 将掩码应用于FFT图像并进行逆FFT
    masked_image = np.real(np.fft.ifft2(np.fft.ifftshift(fft_image * (1 - healthy_mask))))
    masked_image = masked_image * (1 - unhealthy_mask)

    return masked_image


def main(save_videos=True, bias_corrected=False, verbose=0):
    DATASET = "./DATASETS/CancerousDataset/EdinburghDataset"
    patients = os.listdir(DATASET)
    for i in [f"{DATASET}/Anomalous-T1/raw_new", f"{DATASET}/Anomalous-T1/mask_new"]:
        try:
            os.makedirs(i)
        except OSError:
            pass
    if save_videos:
        for i in [f"{DATASET}/Anomalous-T1/raw_new/videos", f"{DATASET}/Anomalous-T1/mask_new/videos"]:
            try:
                os.makedirs(i)
            except OSError:
                pass

    for patient in patients:
        try:
            patient_data = os.listdir(f"{DATASET}/{patient}")
        except:
            if verbose:
                print(f"{DATASET}/{patient} Not a directory")
            continue
        for data_folder in patient_data:
            if "COR_3D" in data_folder:
                try:
                    T1_files = os.listdir(f"{DATASET}/{patient}/{data_folder}")
                except:
                    if verbose:
                        print(f"{patient}/{data_folder} not a directory")
                    continue
                try:
                    mask_dir = os.listdir(f"{DATASET}/{patient}/tissue_classes")
                    for file in mask_dir:
                        if file.startswith("cleaned") and file.endswith(".nii"):
                            mask_file = file
                except:
                    if verbose:
                        print(f"{DATASET}/{patient}/tissue_classes dir not found")
                    return
                for t1 in T1_files:
                    if bias_corrected:
                        check = t1.endswith("corrected.nii")
                    else:
                        check = t1.startswith("anon")
                    if check and t1.endswith(".nii"):
                        # try:
                        # use slice 35-55
                        img = nib.load(f"{DATASET}/{patient}/{data_folder}/{t1}")
                        mask = nib.load(f"{DATASET}/{patient}/tissue_classes/{mask_file}").get_fdata()
                        image = img.get_fdata()
                        if verbose:
                            print(image.shape)
                        if bias_corrected:
                            # image.shape = (256, 156, 256)
                            image = np.rot90(image, 3, (0, 2))
                            image = np.flip(image, 1)
                            # image.shape = (256, 156, 256)
                        else:
                            image = np.transpose(image, (1, 2, 0))
                        mask = np.transpose(mask, (1, 2, 0))
                        if verbose:
                            print(image.shape)
                        image_mean = np.mean(image)
                        image_std = np.std(image)
                        img_range = (image_mean - 1 * image_std, image_mean + 2 * image_std)
                        image = np.clip(image, img_range[0], img_range[1])
                        image = image / (img_range[1] - img_range[0])

                        np.save(
                            f"{DATASET}/Anomalous-T1/raw_new/{patient}.npy", image.astype(
                                np.float32
                            )
                        )
                        np.save(
                            f"{DATASET}/Anomalous-T1/mask_new/{patient}.npy", mask.astype(
                                np.float32
                            )
                        )
                        if verbose:
                            print(f"Saved {DATASET}/Anomalous-T1/mask/{patient}.npy")

                        if save_videos:
                            fig = plt.figure()
                            ims = []
                            for i in range(image.shape[0]):
                                tempImg = image[i:i + 1, :, :]
                                im = plt.imshow(
                                    tempImg.reshape(image.shape[1], image.shape[2]), cmap='gray', animated=True
                                )
                                ims.append([im])

                            ani = animation.ArtistAnimation(
                                fig, ims, interval=50, blit=True,
                                repeat_delay=1000
                            )

                            ani.save(f"{DATASET}/Anomalous-T1/raw_new/videos/{patient}.gif", writer='pillow')
                            if verbose:
                                print(f"Saved {DATASET}/Anomalous-T1/raw/videos/{patient}.gif")
                            fig = plt.figure()
                            ims = []
                            for i in range(mask.shape[0]):
                                tempImg = mask[i:i + 1, :, :]
                                im = plt.imshow(
                                    tempImg.reshape(mask.shape[1], mask.shape[2]), cmap='gray', animated=True
                                )
                                ims.append([im])

                            ani = animation.ArtistAnimation(
                                fig, ims, interval=50, blit=True,
                                repeat_delay=1000
                            )

                            ani.save(f"{DATASET}/Anomalous-T1/mask_new/videos/{patient}.gif", writer='pillow')
                            if verbose:
                                print(mask.shape)
                                print(f"Saved {DATASET}/Anomalous-T1/raw/videos/{patient}.gif", writer='pillow')


def checkDataSet():
    resized = False
    mri_dataset = AnomalousMRIDataset(
        "DATASETS/CancerousDataset/EdinburghDataset/Anomalous-T1/raw", img_size=(256, 256),
        slice_selection="iterateUnknown", resized=resized
        # slice_selection="random"
    )

    dataset_loader = cycle(
        torch.utils.data.DataLoader(
            mri_dataset,
            batch_size=22, shuffle=True,
            num_workers=2, drop_last=True
        )
    )

    new = next(dataset_loader)

    image = new["image"]

    print(image.shape)
    from helpers import gridify_output
    print("Making Video for resized =", resized)
    fig = plt.figure()
    ims = []
    for i in range(0, image.shape[1], 2):
        tempImg = image[:, i, ...].reshape(image.shape[0], 1, image.shape[2], image.shape[3])
        im = plt.imshow(
            gridify_output(tempImg, 5), cmap='gray',
            animated=True
        )
        ims.append([im])

    ani = animation.ArtistAnimation(
        fig, ims, interval=50, blit=True,
        repeat_delay=1000
    )

    ani.save(f"./CancerousDataset/EdinburghDataset/Anomalous-T1/video-resized={resized}.gif", writer='pillow')


def output_videos_for_dataset():
    folders = os.listdir("/Users/jules/Downloads/19085/")
    folders.sort()
    print(f"Folders: {folders}")
    for folder in folders:
        try:
            files_folder = os.listdir("/Users/jules/Downloads/19085/" + folder)
        except:
            print(f"{folder} not a folder")
            exit()

        for file in files_folder:
            try:
                if file[-4:] == ".nii":
                    # try:
                    # use slice 35-55
                    img = nib.load("/Users/jules/Downloads/19085/" + folder + "/" + file)
                    image = img.get_fdata()
                    image = np.rot90(image, 3, (0, 2))
                    print(f"{folder}/{file} has shape {image.shape}")
                    outputImg = np.zeros((256, 256, 310))
                    for i in range(image.shape[1]):
                        tempImg = image[:, i:i + 1, :].reshape(image.shape[0], image.shape[2])
                        img_sm = cv2.resize(tempImg, (310, 256), interpolation=cv2.INTER_CUBIC)
                        outputImg[i, :, :] = img_sm

                    image = outputImg
                    print(f"{folder}/{file} has shape {image.shape}")
                    fig = plt.figure()
                    ims = []
                    for i in range(image.shape[0]):
                        tempImg = image[i:i + 1, :, :]
                        im = plt.imshow(tempImg.reshape(image.shape[1], image.shape[2]), cmap='gray', animated=True)
                        ims.append([im])

                    ani = animation.ArtistAnimation(
                        fig, ims, interval=50, blit=True,
                        repeat_delay=1000
                    )

                    ani.save("/Users/jules/Downloads/19085/" + folder + "/" + file + ".gif", writer='pillow')
                    plt.close(fig)

            except:
                print(
                    f"--------------------------------------{folder}/{file} FAILED TO SAVE VIDEO ------------------------------------------------"
                )


def load_datasets_for_test():
    args = {'img_size': (256, 256), 'random_slice': True, 'Batch_Size': 20}
    training, testing = init_datasets("./", args)

    ano_dataset = AnomalousMRIDataset(
        ROOT_DIR=f'DATASETS/CancerousDataset/EdinburghDataset/Anomalous-T1', img_size=args['img_size'],
        slice_selection="random", resized=False
    )

    train_loader = init_dataset_loader(training, args)
    ano_loader = init_dataset_loader(ano_dataset, args)

    for i in range(5):
        new = next(train_loader)
        new_ano = next(ano_loader)
        output = torch.cat((new["image"][:10], new_ano["image"][:10]))
        plt.imshow(helpers.gridify_output(output, 5), cmap='gray')
        plt.show()
        plt.pause(0.0001)


def image_detection_datasets(ROOT_DIR, args):
    dataset = JSONDataset(
        ROOT_DIR=f'{ROOT_DIR}full_vertebrae/patient', img_size=args['img_size'], random_slice=args['random_slice']
    )
    return dataset


def ver_detection_datasets(ROOT_DIR, args):
    if args['arg_num'] == '291' or args['arg_num'] == '301':
        dataset = VERDataset(
            ROOT_DIR=f'{ROOT_DIR}full_vertebrae/patient', img_size=args['img_size'], random_slice=args['random_slice']
        )
    elif args['arg_num'] == '292' or args['arg_num'] == '302':
        dataset = VERDataset(
            ROOT_DIR=f'{ROOT_DIR}crop_full_vertebrae/patient', img_size=args['img_size'],
            random_slice=args['random_slice']
        )
    return dataset


# def detection_diff_seg_datasets(ROOT_DIR, args):
#     if args['arg_num'] == '1':  # detection: diff_seg.py
#         training_dataset = None
#         testing_dataset = diff_seg_Dataset_2(
#             ROOT_DIR=f'/data/hsd/new_vertebrae_data/full_data_1200/processed_images', img_size=args['img_size'],
#             random_slice=args['random_slice']
#         )
#     return training_dataset, testing_dataset
def diff_seg_datasets(ROOT_DIR, args):
    # if args['arg_num'] == '1':
    #     training_dataset = diff_seg_Dataset_1(
    #         ROOT_DIR=f'{ROOT_DIR}data/VerTumor600/MRI_vertebrae/Train', img_size=args['img_size'],
    #         random_slice=args['random_slice']
    #     )
    #     testing_dataset = diff_seg_Dataset_1(
    #         ROOT_DIR=f'{ROOT_DIR}data/VerTumor600/MRI_vertebrae/Test', img_size=args['img_size'],
    #         random_slice=args['random_slice']
    #     )
    if args['arg_num'] == '1':  # detection: diff_seg.py  合成数据  "ex_num": 3
        training_dataset = syn_MRIDataset(
            ROOT_DIR=f'{ROOT_DIR}data/VerTumor600/MRI_vertebrae/Train', img_size=args['img_size'],
            random_slice=args['random_slice']
        )
        testing_dataset = syn_MRIDataset(
            ROOT_DIR=f'{ROOT_DIR}data/VerTumor600/MRI_vertebrae/Test', img_size=args['img_size'],
            random_slice=args['random_slice']
        )
    # elif args['arg_num'] == '2':  # detection: diff_seg.py
    #     training_dataset = diff_seg_Dataset_1(
    #         ROOT_DIR=f'{ROOT_DIR}data/VerTumor600/MRI_vertebrae/Train', img_size=args['img_size'],
    #         random_slice=args['random_slice']
    #     )
    #     testing_dataset = diff_seg_Dataset_1(
    #         ROOT_DIR=f'{ROOT_DIR}data/VerTumor600/MRI_vertebrae/Test', img_size=args['img_size'],
    #         random_slice=args['random_slice']
    #     )
    elif args['arg_num'] == '2':  # detection: generate_map.py
        training_dataset = diff_seg_Dataset_3(
            ROOT_DIR=f'{ROOT_DIR}data/VerTumor600/MRI_vertebrae/Train', img_size=args['img_size'],
            random_slice=args['random_slice']
        )
        testing_dataset = diff_seg_Dataset_3(
            ROOT_DIR=f'{ROOT_DIR}data/VerTumor600/MRI_vertebrae/Test', img_size=args['img_size'],
            random_slice=args['random_slice']
        )
    elif args['arg_num'] == '3':  # Train:resnet_cls,py
        training_dataset = common_classifier_Dataset(
            ROOT_DIR=f'{ROOT_DIR}data/VerTumor600/cropped_vertebrae/Train/image', img_size=args['img_size'],
            random_slice=args['random_slice']
        )
        testing_dataset = common_classifier_Dataset(
            ROOT_DIR=f'{ROOT_DIR}data/VerTumor600/cropped_vertebrae/Test/image', img_size=args['img_size'],
            random_slice=args['random_slice']
        )
    # elif args['arg_num'] == '3':  # Train:resnet_18_cls,py
    #     training_dataset = common_classifier_Dataset(
    #         ROOT_DIR=f'/data/hsd/AnoDDPM/cropped_vertebrae/Train', img_size=args['img_size'],
    #         random_slice=args['random_slice']
    #     )
    #     testing_dataset = common_classifier_Dataset(
    #         ROOT_DIR=f'/data/hsd/AnoDDPM/cropped_vertebrae/Test', img_size=args['img_size'],
    #         random_slice=args['random_slice']
    #     )
    elif args['arg_num'] == '4':  # detection:resnet_detection.py
        # testing_dataset = common_classifier_Dataset_1200(
        #     ROOT_DIR=f'/data/hsd/new_vertebrae_data/full_data_1200/cropped_images', img_size=args['img_size'],
        #     random_slice=args['random_slice']
        # )  # 只有图片没有标签
        # testing_dataset = common_classifier_Dataset_1200(
        #     ROOT_DIR=f'{ROOT_DIR}data/VerTumor600/cropped_vertebrae/Test/image', img_size=args['img_size'],
        #     random_slice=args['random_slice']
        # )  # 有标签了
        testing_dataset = common_detection_Dataset(
            ROOT_DIR=f'{ROOT_DIR}data/VerTumor600/cropped_vertebrae/Test/image', img_size=args['img_size'],
            random_slice=args['random_slice']
        )
        training_dataset = None

    elif args['arg_num'] == '295':
        training_dataset = diff_seg_Dataset(
            ROOT_DIR=f'{ROOT_DIR}diff_seg_vertebrae/Train/', img_size=args['img_size'],
            random_slice=args['random_slice']
        )
        testing_dataset = diff_seg_Dataset(
            ROOT_DIR=f'{ROOT_DIR}diff_seg_vertebrae/Test/', img_size=args['img_size'], random_slice=args['random_slice']
        )

    else:
        training_dataset = diff_seg_Dataset_22(
            ROOT_DIR=f'{ROOT_DIR}vertebrae_mask_unhealthy/Train/', img_size=args['img_size'],
            random_slice=args['random_slice']
        )
        testing_dataset = diff_seg_Dataset_22(
            ROOT_DIR=f'{ROOT_DIR}vertebrae_mask_unhealthy/Test/', img_size=args['img_size'],
            random_slice=args['random_slice']
        )
    return training_dataset, testing_dataset


def crop_datasets(ROOT_DIR, args):
    training_dataset = diff_seg_Dataset(
        ROOT_DIR=f'{ROOT_DIR}cropped_vertebrae/Train/', img_size=args['img_size'], random_slice=args['random_slice']
    )
    testing_dataset = diff_seg_Dataset(
        ROOT_DIR=f'{ROOT_DIR}cropped_vertebrae/Test/', img_size=args['img_size'], random_slice=args['random_slice']
    )
    return training_dataset, testing_dataset


def crop_classifier_datasets(ROOT_DIR, args):
    if args['arg_num'] == '21':
        training_dataset = common_classifier_Dataset(
            ROOT_DIR=f'{ROOT_DIR}cropped_vertebrae/Train/', img_size=args['img_size'], random_slice=args['random_slice']
        )
        testing_dataset = common_classifier_Dataset(
            ROOT_DIR=f'{ROOT_DIR}cropped_vertebrae/Test41/', img_size=args['img_size'],
            random_slice=args['random_slice']
        )
    elif args['arg_num'] == '3':
        training_dataset = common_classifier_Dataset(
            ROOT_DIR=f'{ROOT_DIR}data/cropped_vertebrae/Train/', img_size=args['img_size'],
            random_slice=args['random_slice']
        )
        testing_dataset = common_classifier_Dataset(
            ROOT_DIR=f'{ROOT_DIR}data/cropped_vertebrae/Test/', img_size=args['img_size'],
            random_slice=args['random_slice']
        )
    elif args['arg_num'] == '263' or args['arg_num'] == '264':
        training_dataset = classifier_Dataset_200(
            ROOT_DIR=f'{ROOT_DIR}cropped_vertebrae/Train/', img_size=args['img_size'], random_slice=args['random_slice']
        )
        testing_dataset = classifier_Dataset(
            ROOT_DIR=f'{ROOT_DIR}cropped_vertebrae/Test41/', img_size=args['img_size'],
            random_slice=args['random_slice']
        )
    elif args['arg_num'] == '4':
        training_dataset = classifier_Dataset(
            ROOT_DIR=f'{ROOT_DIR}cropped_vertebrae/Train/', img_size=args['img_size'], random_slice=args['random_slice']
        )
        testing_dataset = classifier_Dataset(
            ROOT_DIR=f'{ROOT_DIR}cropped_vertebrae/Test/', img_size=args['img_size'],
            random_slice=args['random_slice']
        )
    else:
        training_dataset = classifier_Dataset(
            ROOT_DIR=f'{ROOT_DIR}cropped_vertebrae/Train/', img_size=args['img_size'], random_slice=args['random_slice']
        )
        testing_dataset = classifier_Dataset(
            ROOT_DIR=f'{ROOT_DIR}cropped_vertebrae/Test41/', img_size=args['img_size'],
            random_slice=args['random_slice']
        )
    return training_dataset, testing_dataset


def balanced_crop_classifier_datasets(ROOT_DIR, args):
    training_hea_dataset = common_classifier_Dataset(
        ROOT_DIR=f'{ROOT_DIR}balanced_cropped_vertebrae/Train_healthy/', img_size=args['img_size'],
        random_slice=args['random_slice']
    )
    training_unhea_dataset = common_classifier_Dataset(
        ROOT_DIR=f'{ROOT_DIR}balanced_cropped_vertebrae/Train_unhealthy/', img_size=args['img_size'],
        random_slice=args['random_slice']
    )
    testing_hea_dataset = common_classifier_Dataset(
        ROOT_DIR=f'{ROOT_DIR}balanced_cropped_vertebrae/Test_healthy/', img_size=args['img_size'],
        random_slice=args['random_slice']
    )
    testing_unhea_dataset = common_classifier_Dataset(
        ROOT_DIR=f'{ROOT_DIR}balanced_cropped_vertebrae/Test_unhealthy/', img_size=args['img_size'],
        random_slice=args['random_slice']
    )
    return training_hea_dataset, training_unhea_dataset, testing_hea_dataset, testing_unhea_dataset


def init_datasets(ROOT_DIR, args):
    if args['arg_num'] == '291' or args['arg_num'] == '301':
        training_dataset = MRIDataset(
            ROOT_DIR=f'{ROOT_DIR}full_vertebrae/Train/', img_size=args['img_size'], random_slice=args['random_slice']
        )
        testing_dataset = MRIDataset(
            ROOT_DIR=f'{ROOT_DIR}full_vertebrae/Test/', img_size=args['img_size'], random_slice=args['random_slice']
        )
    elif args['arg_num'] == '292' or args['arg_num'] == '302':
        # hy-tmp/AnoDDPM/crop_full_vertebrace
        training_dataset = MRIDataset(
            ROOT_DIR=f'{ROOT_DIR}crop_full_vertebrae/Train/', img_size=args['img_size'],
            random_slice=args['random_slice']
        )
        testing_dataset = MRIDataset(
            ROOT_DIR=f'{ROOT_DIR}crop_full_vertebrae/Test/', img_size=args['img_size'],
            random_slice=args['random_slice']
        )
    # elif args['arg_num'] == '299':

    return training_dataset, testing_dataset


def syn_datasets(ROOT_DIR, args):
    # if args['arg_num'] == '293' :
    training_dataset = syn_MRIDataset(
        ROOT_DIR=f'{ROOT_DIR}diff_seg_syn_vertebrae/Train/', img_size=args['img_size'],
        random_slice=args['random_slice']
    )
    testing_dataset = syn_MRIDataset(
        ROOT_DIR=f'{ROOT_DIR}diff_seg_syn_vertebrae/Test/', img_size=args['img_size'], random_slice=args['random_slice']
    )
    return training_dataset, testing_dataset


def init_dataset_loader(mri_dataset, args, shuffle=True):
    dataset_loader = cycle(
        torch.utils.data.DataLoader(
            mri_dataset,
            batch_size=args['Batch_Size'],
            shuffle=shuffle,
            # num_workers=4,
            # sampler=sampler,
            # drop_last=True
        )
    )

    return dataset_loader


def sampler_dataset_loader(healthy_dataset, unhealthy_dataset, args, shuffle=True):
    # 假设已经有健康数据集 healthy_dataset 和不健康数据集 unhealthy_dataset

    # 计算健康和不健康数据的样本数量
    healthy_count = len(healthy_dataset)
    unhealthy_count = len(unhealthy_dataset)

    # 确定健康和不健康数据的比例
    ratio = float(unhealthy_count) / float(healthy_count)

    # 计算每个类别的采样权重
    weights = [1.0, ratio]

    # 创建采样器
    sampler = WeightedRandomSampler(weights, len(weights))

    # 创建新的数据加载器
    # batch_size = 32  # 根据需求设置批次大小
    # balanced_dataloader = DataLoader(
    #     dataset=torch.utils.data.ConcatDataset([healthy_dataset, unhealthy_dataset]),
    #     batch_size=args['Batch_Size'],
    #     sampler=sampler,
    #     drop_last=True)

    # 无限循环的数据加载器
    balanced_dataloader = iter(cycle(
        DataLoader(
            dataset=torch.utils.data.ConcatDataset([healthy_dataset, unhealthy_dataset]),
            batch_size=args['Batch_Size'],
            sampler=sampler,
            drop_last=True)
    ))

    return balanced_dataloader


def over_sample_dataset(healthy_dataset, unhealthy_dataset, over_sampling_factor):
    over_sampled_unhealthy_dataset = []
    unhealthy_count = len(unhealthy_dataset)

    for _ in range(over_sampling_factor):
        for i in range(unhealthy_count):
            unhealthy_sample = unhealthy_dataset[i]
            over_sampled_unhealthy_dataset.append(unhealthy_sample)

    merged_dataset = torch.utils.data.ConcatDataset([healthy_dataset] + over_sampled_unhealthy_dataset)

    return merged_dataset


def over_sampler_dataset_loader(healthy_dataset, unhealthy_dataset, args, shuffle=True):
    # 计算过采样的倍数
    over_sampling_factor = len(healthy_dataset) // len(unhealthy_dataset)

    # 创建过采样的数据集
    merged_dataset = over_sample_dataset(healthy_dataset, unhealthy_dataset, over_sampling_factor)

    # 使用 DataLoader 加载数据
    dataset_loader = DataLoader(
        dataset=merged_dataset,
        batch_size=args['Batch_Size'],
        shuffle=shuffle,
        drop_last=True
    )

    # 创建无限循环迭代器
    dataset_loader = itertools.cycle(dataset_loader)

    return dataset_loader, merged_dataset


class DAGM(Dataset):
    def __init__(self, dir, anomalous=False, img_size=(256, 256), rgb=False, random_crop=True):
        # dir = './DATASETS/Carpet/Class1'
        if anomalous and dir[-4:] != "_def":
            dir += "_def"
        self.ROOT_DIR = dir
        self.anomalous = anomalous
        if rgb:
            norm_const = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        else:
            norm_const = ((0.5), (0.5))

        if random_crop:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(*norm_const)
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize(*norm_const)
                ]
            )
        self.rgb = rgb
        self.img_size = img_size
        self.random_crop = random_crop
        if anomalous:
            self.coord_info = self.load_coordinates(os.path.join(self.ROOT_DIR, "labels.txt"))
        self.filenames = os.listdir(self.ROOT_DIR)
        for i in self.filenames[:]:
            if not i.endswith(".png"):
                self.filenames.remove(i)
        self.filenames = sorted(self.filenames, key=lambda x: int(x[:-4]))

    def load_coordinates(self, path_to_coor):
        '''
        '''

        coord_dict_all = {}
        with open(path_to_coor) as f:
            coordinates = f.read().split('\n')
            for coord in coordinates:
                # print(len(coord.split('\t')))
                if len(coord.split('\t')) == 6:
                    coord_dict = {}
                    coord_split = coord.split('\t')
                    # print(coord_split)
                    # print('\n')
                    coord_dict['major_axis'] = round(float(coord_split[1]))
                    coord_dict['minor_axis'] = round(float(coord_split[2]))
                    coord_dict['angle'] = float(coord_split[3])
                    coord_dict['x'] = round(float(coord_split[4]))
                    coord_dict['y'] = round(float(coord_split[5]))
                    index = int(coord_split[0]) - 1
                    coord_dict_all[index] = coord_dict

        return coord_dict_all

    def make_mask(self, idx, img):
        mask = np.zeros_like(img)
        mask = cv2.ellipse(
            mask,
            (int(self.coord_info[idx]['x']), int(self.coord_info[idx]['y'])),
            (int(self.coord_info[idx]['major_axis']), int(self.coord_info[idx]['minor_axis'])),
            (self.coord_info[idx]['angle'] / 4.7) * 270,
            0,
            360,
            (255, 255, 255),
            -1
        )

        mask[mask > 0] = 255
        return mask

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        # print(repr(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {"filenames": self.filenames[idx]}
        if self.rgb:
            sample["image"] = cv2.imread(os.path.join(self.ROOT_DIR, self.filenames[idx]), 1)
            # sample["image"] = Image.open(os.path.join(self.ROOT_DIR, self.filenames[idx]), "r")
        else:
            sample["image"] = cv2.imread(os.path.join(self.ROOT_DIR, self.filenames[idx]), 0)

        if self.anomalous:
            sample["mask"] = self.make_mask(int(self.filenames[idx][:-4]) - 1, sample["image"])
        if self.random_crop:
            x1 = randint(0, sample["image"].shape[-1] - self.img_size[1])
            y1 = randint(0, sample["image"].shape[-2] - self.img_size[0])
            if self.anomalous:
                sample["mask"] = sample["mask"][x1:x1 + self.img_size[1], y1:y1 + self.img_size[0]]
            sample["image"] = sample["image"][x1:x1 + self.img_size[1], y1:y1 + self.img_size[0]]

        if self.transform:
            image = self.transform(sample["image"])
            if self.anomalous:
                sample["mask"] = self.transform(sample["mask"])
                sample["mask"] = (sample["mask"] > 0).float()
        sample["image"] = image.reshape(1, *self.img_size)

        return sample


class MVTec(Dataset):
    def __init__(self, dir, anomalous=False, img_size=(256, 256), rgb=True, random_crop=True, include_good=False):
        # dir = './DATASETS/leather'

        self.ROOT_DIR = dir
        self.anomalous = anomalous
        if not anomalous:
            self.ROOT_DIR += "/train/good"

        transforms_list = [transforms.ToPILImage()]

        if rgb:
            channels = 3
        else:
            channels = 1
            transforms_list.append(transforms.Grayscale(num_output_channels=channels))
        transforms_mask_list = [transforms.ToPILImage(), transforms.Grayscale(num_output_channels=channels)]
        if not random_crop:
            transforms_list.append(transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR))
            transforms_mask_list.append(transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR))
        transforms_list.append(transforms.ToTensor())
        transforms_mask_list.append(transforms.ToTensor())
        if rgb:
            transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        else:
            transforms_list.append(transforms.Normalize((0.5), (0.5)))
        transforms_mask_list.append(transforms.Normalize((0.5), (0.5)))
        self.transform = transforms.Compose(transforms_list)
        self.transform_mask = transforms.Compose(transforms_mask_list)

        self.rgb = rgb
        self.img_size = img_size
        self.random_crop = random_crop
        self.classes = ["color", "cut", "fold", "glue", "poke"]
        if include_good:
            self.classes.append("good")
        if anomalous:
            self.filenames = [f"{self.ROOT_DIR}/test/{i}/{x}" for i in self.classes for x in
                              os.listdir(self.ROOT_DIR + f"/test/{i}")]

        else:
            self.filenames = [f"{self.ROOT_DIR}/{i}" for i in os.listdir(self.ROOT_DIR)]

        for i in self.filenames[:]:
            if not i.endswith(".png"):
                self.filenames.remove(i)
        self.filenames = sorted(self.filenames, key=lambda x: int(x[-7:-4]))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # print(repr(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {"filenames": self.filenames[idx]}
        if self.rgb:
            sample["image"] = cv2.cvtColor(cv2.imread(os.path.join(self.filenames[idx]), 1), cv2.COLOR_BGR2RGB)
            # sample["image"] = Image.open(os.path.join(self.ROOT_DIR, self.filenames[idx]), "r")
        else:
            sample["image"] = cv2.imread(os.path.join(self.filenames[idx]), 0)
            sample["image"] = sample["image"].reshape(*sample["image"].shape, 1)

        if self.anomalous:
            file = self.filenames[idx].split("/")
            if file[-2] == "good":
                sample["mask"] = np.zeros((sample["image"].shape[0], sample["image"].shape[1], 1)).astype(np.uint8)
            else:
                sample["mask"] = cv2.imread(
                    os.path.join(self.ROOT_DIR, "ground_truth", file[-2], file[-1][:-4] + "_mask.png"), 0
                )
        if self.random_crop:
            x1 = randint(0, sample["image"].shape[-2] - self.img_size[1])
            y1 = randint(0, sample["image"].shape[-3] - self.img_size[0])
            if self.anomalous:
                sample["mask"] = sample["mask"][x1:x1 + self.img_size[1], y1:y1 + self.img_size[0]]
            sample["image"] = sample["image"][x1:x1 + self.img_size[1], y1:y1 + self.img_size[0]]

        if self.transform:
            sample["image"] = self.transform(sample["image"])
            if self.anomalous:
                sample["mask"] = self.transform_mask(sample["mask"])
                sample["mask"] = (sample["mask"] > 0).float()

        return sample


class JSONDataset(Dataset):
    """my ver dataset."""

    def __init__(self, ROOT_DIR, transform=None, img_size=(32, 32), random_slice=False):
        """
        Args:
            ROOT_DIR (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # if img_size==(256,256):
        self.image_size = img_size
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             # 下面的img_size是传进来的参数
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))
             ]
        ) if not transform else transform
        self.mask_transform = transforms.Compose(
            [transforms.ToPILImage(),
             # 下面的img_size是传进来的参数
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             ]
        ) if not transform else transform

        # self.filenames = os.listdir(ROOT_DIR)  # 返回ROOT_DIR下 所有子目录的名字 的列表

        # if ".DS_Store" in self.filenames:
        #     self.filenames.remove(".DS_Store")
        # if ".ipynb_checkpoints" in self.filenames:
        #     self.filenames.remove(".ipynb_checkpoints")

        self.ROOT_DIR = ROOT_DIR
        self.random_slice = random_slice

        with open("../NEW/detection_ver200204.json", "r") as f:
            s = json.load(f)

        self.images = s['images']
        self.images = np.array(self.images)  # 刚才是list，现在变成array，应该是(135, 512, 512)的np.array。

        self.masks = s['masks']
        self.masks = np.array(self.masks)  # 应该也是(135, 512, 512)的np.array。

        self.cancer_diag = s['cancer_diag']

        self.cancer_diag_sus = s['cancer_diag_sus']

        self.all_patients = self.images.shape[0]  # 一共多少张图。

        self.cancer = []

        def get_id_list(cancer_temp):
            id_list = []
            classes = ["", "S1", "L5", "L4", "L3", "L2", "L1", "T12", "T11", "T10"]
            for temp in cancer_temp:
                if temp == 'S':
                    temp = 'S1'
                id_list.append(classes.index(temp))

            return id_list

        for p in range(self.all_patients):
            temp_cancer = []

            temp1 = self.cancer_diag[p]
            temp2 = self.cancer_diag_sus[p]

            for i in temp1:
                temp_cancer.append(i)

            for j in temp2:
                if j not in temp_cancer:
                    temp_cancer.append(j)

            if len(temp_cancer) > 1 and '' in temp_cancer:
                temp_cancer.remove('')

            cancer_number = get_id_list(temp_cancer)

            self.cancer.append(cancer_number)

        # self.cancer = np.array(self.cancer)

    def __len__(self):
        return self.all_patients

    def __getitem__(self, idx):
        # print(repr(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx].astype(np.uint8)
        mask = self.masks[idx].astype(np.uint8)
        cancer_id = self.cancer[idx]
        cancer_id = np.array(cancer_id)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # 转换为灰度图

        # threshold_value = 128
        # max_value = 255
        # _, mask = cv2.threshold(mask, threshold_value, max_value, cv2.THRESH_BINARY)

        # image = cv2.equalizeHist(image)  # 直方图均衡化

        if self.transform:
            # (256,192)->(256,256)
            # image = np.array(image)
            image = self.transform(image)
            mask = self.mask_transform(mask)

        sample = {'image': image,
                  'mask': mask,
                  'mask_id': self.masks[idx],
                  'cancer_id': cancer_id,
                  "filenames": idx}
        return sample


class JSONDataset_2(Dataset):
    """my ver dataset."""

    def __init__(self, ROOT_DIR, random_slice=False):

        self.filenames = os.listdir(ROOT_DIR)  # 返回ROOT_DIR下 所有子目录的名字 的列表
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        if ".ipynb_checkpoints" in self.filenames:
            self.filenames.remove(".ipynb_checkpoints")
        self.ROOT_DIR = ROOT_DIR
        self.random_slice = random_slice

        with open("detection_ver200204.json", "r") as f:
            s = json.load(f)

        self.images = s['images']
        self.images = np.array(self.images)  # 刚才是list，现在变成array，应该是(135, 512, 512)的np.array。

        self.masks = s['masks']
        self.masks = np.array(self.masks)  # 应该也是(135, 512, 512)的np.array。

        cancer_diag = s['cancer_diag']

        cancer_diag_sus = s['cancer_diag_sus']

        all_patients = self.images.shape[0]  # 一共多少张图。

        self.cancer = []

        for p in range(all_patients):
            temp_cancer = []

            temp1 = cancer_diag[p]
            temp2 = cancer_diag_sus[p]

            for i in temp1:
                temp_cancer.append(i)

            for j in temp2:
                if j not in temp_cancer:
                    temp_cancer.append(j)

            if len(temp_cancer) > 1 and '' in temp_cancer:
                temp_cancer.remove('')

            self.cancer.append(temp_cancer)

        def get_id_list(cancer_temp):
            id_list = []
            classes = ["S1", "L5", "L4", "L3", "L2", "L1", "T12", "T11", "T10"]
            for temp in cancer_temp:
                if temp == 'S':
                    temp = 'S1'
                id_list.append(classes.index(temp) + 1)

            return id_list

        self.cancer_number = get_id_list(self.cancer)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'image': self.images[idx],
                  'mask': self.masks[idx],
                  'mask_id': self.masks[idx],
                  'cancer_id': self.cancer_number[idx]}
        return sample


class VERDataset(Dataset):
    """my ver dataset."""

    def __init__(self, ROOT_DIR, transform=None, img_size=(32, 32), random_slice=False):
        """
        Args:
            ROOT_DIR (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        '''对于预处理，
        我们应用 ±3° 的随机旋转和 0.02×width 和 0.09×height 的随机平移，
        然后中心裁剪 235，调整为 256×256。'''
        # if img_size==(256,256):
        self.image_size = img_size
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             # 下面的img_size是传进来的参数
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))
             ]
        ) if not transform else transform

        self.filenames = os.listdir(ROOT_DIR)  # 返回ROOT_DIR下 所有子目录的名字 的列表
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        if ".ipynb_checkpoints" in self.filenames:
            self.filenames.remove(".ipynb_checkpoints")
        self.ROOT_DIR = ROOT_DIR
        self.random_slice = random_slice

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # print(repr(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "image.jpg")
        unhealthy_mask_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "unhealthy_mask.jpg")
        square_mask_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "square_mask.jpg")
        vertebrae_mask_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "vertebrae_mask.jpg")

        image = cv2.imread(img_name)
        unhealthy_mask = cv2.imread(unhealthy_mask_name)
        square_mask = cv2.imread(square_mask_name)
        vertebrae_mask = cv2.imread(vertebrae_mask_name)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        unhealthy_mask = cv2.cvtColor(unhealthy_mask, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        square_mask = cv2.cvtColor(square_mask, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        vertebrae_mask = cv2.cvtColor(vertebrae_mask, cv2.COLOR_BGR2GRAY)  # 转换为灰度图

        # threshold_value = 128
        # max_value = 255
        # _, mask = cv2.threshold(mask, threshold_value, max_value, cv2.THRESH_BINARY)

        # image = cv2.equalizeHist(image)  # 直方图均衡化

        if self.transform:
            # (256,192)->(256,256)
            # image = np.array(image)
            image = self.transform(image)
            unhealthy_mask = self.transform(unhealthy_mask)
            square_mask = self.transform(square_mask)
            vertebrae_mask = self.transform(vertebrae_mask)

        sample = {'image': image,
                  'unhealthy_mask': unhealthy_mask,
                  'square_mask': square_mask,
                  'vertebrae_mask': vertebrae_mask,
                  "filenames": self.filenames[idx]}
        return sample


class classifier_Dataset(Dataset):
    """my ver dataset."""

    def __init__(self, ROOT_DIR, transform=None, img_size=(32, 32), random_slice=False):
        """
        Args:
            ROOT_DIR (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # if img_size==(256,256):
        self.image_size = img_size
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             # 下面的img_size是传进来的参数
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))
             ]
        ) if not transform else transform
        self.mask_transform = transforms.Compose(
            [transforms.ToPILImage(),
             # 下面的img_size是传进来的参数
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             ]
        ) if not transform else transform
        self.filenames = os.listdir(ROOT_DIR)  # 返回ROOT_DIR下 所有子目录的名字 的列表
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        if ".ipynb_checkpoints" in self.filenames:
            self.filenames.remove(".ipynb_checkpoints")
        self.ROOT_DIR = ROOT_DIR
        self.random_slice = random_slice

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # print(repr(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()

        save_dir = os.path.join(self.ROOT_DIR, self.filenames[idx])
        img_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "image.png")
        unhealthy_mask_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "mask.png")
        diffusion_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "diffusion.png")
        out_mask_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "out_mask.png")

        diffusion_name_idx = []
        for idx in range(1, 11):
            name_idx = os.path.join(self.ROOT_DIR, self.filenames[idx], "diffusion_" + str(idx * 10) + ".png")
            diffusion_name_idx.append(name_idx)

        image = cv2.imread(img_name)
        unhealthy_mask = cv2.imread(unhealthy_mask_name)
        diff = cv2.imread(diffusion_name)
        out_mask = cv2.imread(out_mask_name)
        diffusion_10 = [cv2.imread(d) for d in diffusion_name_idx]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        unhealthy_mask = cv2.cvtColor(unhealthy_mask, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        out_mask = cv2.cvtColor(out_mask, cv2.COLOR_BGR2GRAY)
        diffusion_10_BGR2GRAY = [cv2.cvtColor(d, cv2.COLOR_BGR2GRAY) for d in diffusion_10]

        # threshold_value = 128
        # max_value = 255
        # _, mask = cv2.threshold(mask, threshold_value, max_value, cv2.THRESH_BINARY)

        # image = cv2.equalizeHist(image)  # 直方图均衡化

        if self.transform:
            # (256,192)->(256,256)
            # image = np.array(image)
            image = self.transform(image)
            unhealthy_mask = self.mask_transform(unhealthy_mask)
            diff = self.transform(diff)
            out_mask = self.transform(out_mask)
            diffusion_10_transform = [self.transform(d) for d in diffusion_10_BGR2GRAY]

        if 1 in unhealthy_mask:
            label = 1
        else:
            label = 0

        sample = {'image': image,
                  'label': label,
                  'unhealthy_mask': unhealthy_mask,
                  'diffusion': diff,
                  'out_mask': out_mask,
                  'diffusion_10': diffusion_10_transform,
                  "filenames": self.filenames[idx],
                  'save_dir': save_dir}
        return sample


class classifier_Dataset_200(Dataset):
    """my ver dataset."""

    def __init__(self, ROOT_DIR, transform=None, img_size=(32, 32), random_slice=False):
        """
        Args:
            ROOT_DIR (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # if img_size==(256,256):
        self.image_size = img_size
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             # 下面的img_size是传进来的参数
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))
             ]
        ) if not transform else transform
        self.mask_transform = transforms.Compose(
            [transforms.ToPILImage(),
             # 下面的img_size是传进来的参数
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             ]
        ) if not transform else transform
        self.filenames = os.listdir(ROOT_DIR)  # 返回ROOT_DIR下 所有子目录的名字 的列表
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        if ".ipynb_checkpoints" in self.filenames:
            self.filenames.remove(".ipynb_checkpoints")
        self.ROOT_DIR = ROOT_DIR
        self.random_slice = random_slice

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # print(repr(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()

        save_dir = os.path.join(self.ROOT_DIR, self.filenames[idx])
        img_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "image.png")
        unhealthy_mask_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "mask.png")
        diffusion_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "diffusion.png")
        out_mask_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "out_mask.png")

        diffusion_name_idx = []
        for idx in range(1, 21):
            name_idx = os.path.join(self.ROOT_DIR, self.filenames[idx], "diffusion_" + str(idx * 10) + ".png")
            diffusion_name_idx.append(name_idx)

        image = cv2.imread(img_name)
        unhealthy_mask = cv2.imread(unhealthy_mask_name)
        diff = cv2.imread(diffusion_name)
        out_mask = cv2.imread(out_mask_name)
        diffusion_10 = [cv2.imread(d) for d in diffusion_name_idx]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        unhealthy_mask = cv2.cvtColor(unhealthy_mask, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        out_mask = cv2.cvtColor(out_mask, cv2.COLOR_BGR2GRAY)
        diffusion_10_BGR2GRAY = [cv2.cvtColor(d, cv2.COLOR_BGR2GRAY) for d in diffusion_10]

        # threshold_value = 128
        # max_value = 255
        # _, mask = cv2.threshold(mask, threshold_value, max_value, cv2.THRESH_BINARY)

        # image = cv2.equalizeHist(image)  # 直方图均衡化

        if self.transform:
            # (256,192)->(256,256)
            # image = np.array(image)
            image = self.transform(image)
            unhealthy_mask = self.mask_transform(unhealthy_mask)
            diff = self.transform(diff)
            out_mask = self.transform(out_mask)
            diffusion_10_transform = [self.transform(d) for d in diffusion_10_BGR2GRAY]

        if 1 in unhealthy_mask:
            label = 1
        else:
            label = 0

        sample = {'image': image,
                  'label': label,
                  'unhealthy_mask': unhealthy_mask,
                  'diffusion': diff,
                  'out_mask': out_mask,
                  'diffusion_20': diffusion_10_transform,
                  "filenames": self.filenames[idx],
                  'save_dir': save_dir}
        return sample


class common_classifier_Dataset(Dataset):
    """my ver dataset."""

    def __init__(self, ROOT_DIR, transform=None, img_size=(32, 32), random_slice=False):
        """
        Args:
            ROOT_DIR (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # if img_size==(256,256):
        self.image_size = img_size
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             # 下面的img_size是传进来的参数
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))
             ]
        ) if not transform else transform
        self.mask_transform = transforms.Compose(
            [transforms.ToPILImage(),
             # 下面的img_size是传进来的参数
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             ]
        ) if not transform else transform
        self.filenames = os.listdir(ROOT_DIR)  # 返回ROOT_DIR下 所有子目录的名字 的列表
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        if ".ipynb_checkpoints" in self.filenames:
            self.filenames.remove(".ipynb_checkpoints")
        self.ROOT_DIR = ROOT_DIR
        self.random_slice = random_slice

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # print(repr(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()

        save_dir = os.path.join(self.ROOT_DIR, self.filenames[int(idx)])

        # save_dir = os.path.join(self.ROOT_DIR, self.filenames[idx])
        img_name = os.path.join(self.ROOT_DIR, self.filenames[int(idx)], "image.png")
        # unhealthy_mask_name = os.path.join(self.ROOT_DIR, self.filenames[int(idx)], "mask.png")
        diffusion_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "rec.png")
        out_mask_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "out_mask.png")

        # diffusion_name_idx = []
        # for idx in range(1, 11):
        #     name_idx = os.path.join(self.ROOT_DIR, self.filenames[idx], "diffusion_" + str(idx * 10) + ".png")
        #     diffusion_name_idx.append(name_idx)

        image = cv2.imread(img_name)
        # unhealthy_mask = cv2.imread(unhealthy_mask_name)
        diff = cv2.imread(diffusion_name)
        out_mask = cv2.imread(out_mask_name)
        # diffusion_10 = [cv2.imread(d) for d in diffusion_name_idx]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        # unhealthy_mask = cv2.cvtColor(unhealthy_mask, cv2.COLOR_BGR2GRAY)  # 转换为灰度图

        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        out_mask = cv2.cvtColor(out_mask, cv2.COLOR_BGR2GRAY)
        # diffusion_10_BGR2GRAY = [cv2.cvtColor(d, cv2.COLOR_BGR2GRAY) for d in diffusion_10]

        # threshold_value = 128
        # max_value = 255
        # _, mask = cv2.threshold(mask, threshold_value, max_value, cv2.THRESH_BINARY)

        # image = cv2.equalizeHist(image)  # 直方图均衡化

        if self.transform:
            # (256,192)->(256,256)
            # image = np.array(image)
            image = self.transform(image)
            # unhealthy_mask = self.mask_transform(unhealthy_mask)
            diff = self.transform(diff)
            out_mask = self.transform(out_mask)
            # diffusion_10_transform = [self.transform(d) for d in diffusion_10_BGR2GRAY]

        if "unhealthy" in save_dir:
            label = 1
        else:
            label = 0

        sample = {'image': image,
                  'label': label,
                  'diff': diff,
                  'out_mask': out_mask,
                  # 'unhealthy_mask': unhealthy_mask,
                  "filenames": self.filenames[int(idx)],
                  'save_dir': save_dir
                  }
        return sample

class common_detection_Dataset(Dataset):
    """my ver dataset."""

    def __init__(self, ROOT_DIR, transform=None, img_size=(32, 32), random_slice=False):
        """
        Args:
            ROOT_DIR (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # if img_size==(256,256):
        self.image_size = img_size
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             # 下面的img_size是传进来的参数
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))
             ]
        ) if not transform else transform
        self.mask_transform = transforms.Compose(
            [transforms.ToPILImage(),
             # 下面的img_size是传进来的参数
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             ]
        ) if not transform else transform
        self.filenames = os.listdir(ROOT_DIR)  # 返回ROOT_DIR下 所有子目录的名字 的列表
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        if ".ipynb_checkpoints" in self.filenames:
            self.filenames.remove(".ipynb_checkpoints")
        self.ROOT_DIR = ROOT_DIR
        self.random_slice = random_slice

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # print(repr(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()

        save_dir = os.path.join(self.ROOT_DIR, self.filenames[int(idx)])
        # img_name = os.path.join(self.ROOT_DIR, self.filenames[int(idx)], "image.png")
        image = cv2.imread(save_dir)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图

        if self.transform:
            image = self.transform(image)


        if "unhealthy" in save_dir:
            label = 1
        else:
            label = 0

        sample = {'image': image,
                  'label': label,
                  "filenames": self.filenames[int(idx)],
                  'save_dir': save_dir
                  }
        return sample
class common_classifier_Dataset_1200(Dataset):
    """my ver dataset."""

    def __init__(self, ROOT_DIR, transform=None, img_size=(32, 32), random_slice=False):
        """
        Args:
            ROOT_DIR (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # if img_size==(256,256):
        self.image_size = img_size
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             # 下面的img_size是传进来的参数
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))
             ]
        ) if not transform else transform
        self.mask_transform = transforms.Compose(
            [transforms.ToPILImage(),
             # 下面的img_size是传进来的参数
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             ]
        ) if not transform else transform
        self.filenames = os.listdir(ROOT_DIR)  # 返回ROOT_DIR下 所有子目录的名字 的列表
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        if ".ipynb_checkpoints" in self.filenames:
            self.filenames.remove(".ipynb_checkpoints")
        self.ROOT_DIR = ROOT_DIR
        self.random_slice = random_slice

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # print(repr(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.ROOT_DIR, self.filenames[int(idx)])
        image = cv2.imread(img_name)
        # unhealthy_mask = cv2.imread(unhealthy_mask_name)
        # diff = cv2.imread(diffusion_name)
        # out_mask = cv2.imread(out_mask_name)
        # diffusion_10 = [cv2.imread(d) for d in diffusion_name_idx]

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        # unhealthy_mask = cv2.cvtColor(unhealthy_mask, cv2.COLOR_BGR2GRAY)  # 转换为灰度图

        # diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # out_mask = cv2.cvtColor(out_mask, cv2.COLOR_BGR2GRAY)
        # diffusion_10_BGR2GRAY = [cv2.cvtColor(d, cv2.COLOR_BGR2GRAY) for d in diffusion_10]

        # threshold_value = 128
        # max_value = 255
        # _, mask = cv2.threshold(mask, threshold_value, max_value, cv2.THRESH_BINARY)

        # image = cv2.equalizeHist(image)  # 直方图均衡化

        if "unhealthy" in self.filenames[int(idx)]:
            label = 1
        else:
            label = 0

        if self.transform:
            image = self.transform(image)

        sample = {'image': image,
                  'name': self.filenames[int(idx)],
                  'label': label}
        return sample

class diff_seg_Dataset(Dataset):
    """my ver dataset."""

    def __init__(self, ROOT_DIR, transform=None, img_size=(32, 32), random_slice=False):
        """
        Args:
            ROOT_DIR (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # if img_size==(256,256):
        self.image_size = img_size
        self.transform = transforms.Compose(
            [
             transforms.ToPILImage(),
             # 下面的img_size是传进来的参数
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))
             ]
        ) if not transform else transform
        self.mask_transform = transforms.Compose(
            [transforms.ToPILImage(),
             # 下面的img_size是传进来的参数
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             ]
        ) if not transform else transform
        self.filenames = os.listdir(ROOT_DIR)  # 返回ROOT_DIR下 所有子目录的名字 的列表
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        if ".ipynb_checkpoints" in self.filenames:
            self.filenames.remove(".ipynb_checkpoints")
        self.ROOT_DIR = ROOT_DIR
        self.random_slice = random_slice

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # print(repr(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()

        save_dir = os.path.join(self.ROOT_DIR, self.filenames[idx])
        img_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "image.png")
        mask_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "mask.png")
        unhealthy_mask_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "unhealthy_mask.png")

        image = cv2.imread(img_name)
        mask = cv2.imread(mask_name)
        unhealthy_mask = cv2.imread(unhealthy_mask_name)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        unhealthy_mask = cv2.cvtColor(unhealthy_mask, cv2.COLOR_BGR2GRAY)  # 转换为灰度图

        # 将 mask 中非零部分的值设置为255
        mask[mask > 0] = 255
        # 将 unhealthy_mask 中非零部分的值设置为255
        unhealthy_mask[unhealthy_mask > 0] = 255
        # image = cv2.equalizeHist(image)  # 直方图均衡化

        # threshold_value = 128
        # max_value = 255
        # _, mask = cv2.threshold(mask, threshold_value, max_value, cv2.THRESH_BINARY)

        # input, box_healthy_mask, box_unhealthy_mask = get_IPM_FPM_image(image, "Train", self.filenames[idx])

        if self.transform:
            # (256,192)->(256,256)
            # image = np.array(image)
            image = self.transform(image)
            mask = self.mask_transform(mask)
            unhealthy_mask = self.mask_transform(unhealthy_mask)
            # input = self.transform(input)
            # box_healthy_mask = self.mask_transform(box_healthy_mask)
            # box_unhealthy_mask = self.mask_transform(box_unhealthy_mask)


        sample = {'image': image,
                  # 'input': input,
                  # 'box_healthy_mask': box_healthy_mask,
                  # 'box_unhealthy_mask': box_unhealthy_mask,
                  'mask': mask,
                  'unhealthy_mask': unhealthy_mask,
                  "id": self.filenames[idx],
                  "save_dir": save_dir
                  }
        return sample
class diff_seg_Dataset_1(Dataset):
    """my ver dataset."""

    def __init__(self, ROOT_DIR, transform=None, img_size=(32, 32), random_slice=False):
        """
        Args:
            ROOT_DIR (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # if img_size==(256,256):
        self.image_size = img_size
        self.transform = transforms.Compose(
            [
             transforms.ToPILImage(),
             # 下面的img_size是传进来的参数
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))
             ]
        ) if not transform else transform
        self.mask_transform = transforms.Compose(
            [transforms.ToPILImage(),
             # 下面的img_size是传进来的参数
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             ]
        ) if not transform else transform
        self.filenames = os.listdir(ROOT_DIR)  # 返回ROOT_DIR下 所有子目录的名字 的列表
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        if ".ipynb_checkpoints" in self.filenames:
            self.filenames.remove(".ipynb_checkpoints")
        self.ROOT_DIR = ROOT_DIR
        self.random_slice = random_slice

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # print(repr(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()

        save_dir = os.path.join(self.ROOT_DIR, self.filenames[idx])
        img_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "image.png")
        mask_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "mask.png")
        unhealthy_mask_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "unhealthy_mask.png")

        image = cv2.imread(img_name)
        mask = cv2.imread(mask_name)
        unhealthy_mask = cv2.imread(unhealthy_mask_name)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        unhealthy_mask = cv2.cvtColor(unhealthy_mask, cv2.COLOR_BGR2GRAY)  # 转换为灰度图

        # 将 mask 中非零部分的值设置为255
        mask[mask > 0] = 255
        # 将 unhealthy_mask 中非零部分的值设置为255
        unhealthy_mask[unhealthy_mask > 0] = 255
        # image = cv2.equalizeHist(image)  # 直方图均衡化

        # threshold_value = 128
        # max_value = 255
        # _, mask = cv2.threshold(mask, threshold_value, max_value, cv2.THRESH_BINARY)

        input, box_healthy_mask, box_unhealthy_mask = get_IPM_FPM_image(image, "Train", self.filenames[idx])

        if self.transform:
            # (256,192)->(256,256)
            # image = np.array(image)
            image = self.transform(image)
            mask = self.mask_transform(mask)
            unhealthy_mask = self.mask_transform(unhealthy_mask)
            input = self.transform(input)
            box_healthy_mask = self.mask_transform(box_healthy_mask)
            box_unhealthy_mask = self.mask_transform(box_unhealthy_mask)


        sample = {'image': image,
                  'input': input,
                  'box_healthy_mask': box_healthy_mask,
                  'box_unhealthy_mask': box_unhealthy_mask,
                  'mask': mask,
                  'unhealthy_mask': unhealthy_mask,
                  "id": self.filenames[idx],
                  "save_dir": save_dir
                  }
        return sample


class diff_seg_Dataset_3(Dataset):
    """my ver dataset."""

    def __init__(self, ROOT_DIR, transform=None, img_size=(32, 32), random_slice=False):
        """
        Args:
            ROOT_DIR (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # if img_size==(256,256):
        self.image_size = img_size
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             # 下面的img_size是传进来的参数
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))
             ]
        ) if not transform else transform
        self.mask_transform = transforms.Compose(
            [transforms.ToPILImage(),
             # 下面的img_size是传进来的参数
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             ]
        ) if not transform else transform
        self.filenames = os.listdir(ROOT_DIR)  # 返回ROOT_DIR下 所有子目录的名字 的列表
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        if ".ipynb_checkpoints" in self.filenames:
            self.filenames.remove(".ipynb_checkpoints")
        self.ROOT_DIR = ROOT_DIR
        self.random_slice = random_slice

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # print(repr(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()

        save_dir = os.path.join(self.ROOT_DIR, self.filenames[idx])
        img_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "image.png")
        mask_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "mask.png")
        unhealthy_mask_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "unhealthy_mask.png")

        image = cv2.imread(img_name)
        mask = cv2.imread(mask_name)
        unhealthy_mask = cv2.imread(unhealthy_mask_name)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        unhealthy_mask = cv2.cvtColor(unhealthy_mask, cv2.COLOR_BGR2GRAY)  # 转换为灰度图

        # threshold_value = 128
        # max_value = 255
        # _, mask = cv2.threshold(mask, threshold_value, max_value, cv2.THRESH_BINARY)

        # image = cv2.equalizeHist(image)  # 直方图均衡化

        if self.transform:
            # (256,192)->(256,256)
            # image = np.array(image)
            image = self.transform(image)
            mask = self.mask_transform(mask)
            unhealthy_mask = self.mask_transform(unhealthy_mask)

        sample = {'image': image,
                  'mask': mask,
                  'unhealthy_mask': unhealthy_mask,
                  "id": self.filenames[idx],
                  "save_dir": save_dir
                  }
        return sample


class diff_seg_Dataset_2(Dataset):
    """my ver dataset."""

    def __init__(self, ROOT_DIR, transform=None, img_size=(32, 32), random_slice=False):
        """
        Args:
            ROOT_DIR (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # if img_size==(256,256):
        self.image_size = img_size
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             # 下面的img_size是传进来的参数
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))
             ]
        ) if not transform else transform
        self.RGB_transform = transforms.Compose(
            [transforms.ToPILImage(),
             # 下面的img_size是传进来的参数
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             ]
        ) if not transform else transform
        self.mask_transform = transforms.Compose(
            [transforms.ToPILImage(),
             # 下面的img_size是传进来的参数
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             ]
        ) if not transform else transform
        self.filenames = os.listdir(ROOT_DIR)  # 返回ROOT_DIR下 所有子目录的名字 的列表
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        if ".ipynb_checkpoints" in self.filenames:
            self.filenames.remove(".ipynb_checkpoints")
        self.ROOT_DIR = ROOT_DIR
        self.random_slice = random_slice

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # print(repr(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()

        save_dir = os.path.join(self.ROOT_DIR, self.filenames[idx])
        img_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "image.png")
        mask_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "mask.png")
        with_boxes_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "with_boxes.png")
        # unhealthy_mask_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "unhealthy_mask.png")
        # diffusion_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "diffusion_100.png")
        # out_mask_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "out_mask.png")

        # diffusion_name_idx = []
        # for idx in range(1, 11):
        #     name_idx = os.path.join(self.ROOT_DIR, self.filenames[idx], "diffusion_" + str(idx * 10) + ".png")
        #     diffusion_name_idx.append(name_idx)

        image = cv2.imread(img_name)
        image_with_boxes = cv2.imread(with_boxes_name)
        mask = cv2.imread(mask_name)
        # unhealthy_mask = cv2.imread(unhealthy_mask_name)
        # diff = cv2.imread(diffusion_name)
        # out_mask = cv2.imread(out_mask_name)
        # # diffusion_10 = [cv2.imread(d) for d in diffusion_name_idx]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        # unhealthy_mask = cv2.cvtColor(unhealthy_mask, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        # diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # out_mask = cv2.cvtColor(out_mask, cv2.COLOR_BGR2GRAY)
        # # diffusion_10_BGR2GRAY = [cv2.cvtColor(d, cv2.COLOR_BGR2GRAY) for d in diffusion_10]

        # threshold_value = 128
        # max_value = 255
        # _, mask = cv2.threshold(mask, threshold_value, max_value, cv2.THRESH_BINARY)

        # image = cv2.equalizeHist(image)  # 直方图均衡化

        if self.transform:
            # (256,192)->(256,256)
            # image = np.array(image)
            image = self.transform(image)
            image_with_boxes = self.RGB_transform(image_with_boxes)
            mask = self.mask_transform(mask)
            # unhealthy_mask = self.mask_transform(unhealthy_mask)
            # diff = self.transform(diff)
            # out_mask = self.transform(out_mask)
            # # diffusion_10_transform = [self.transform(d) for d in diffusion_10_BGR2GRAY]

        # if 1 in unhealthy_mask:
        #     label = 1
        # else:
        #     label = 0

        sample = {'image': image,
                  # 'label': label,
                  "image_with_boxes": image_with_boxes,
                  'mask': mask,
                  # 'unhealthy_mask': unhealthy_mask,
                  # 'diffusion': diff,
                  # 'out_mask': out_mask,
                  # 'diffusion_10': diffusion_10_transform,
                  "filenames": self.filenames[idx],
                  'save_dir': save_dir
                  }
        return sample


class diff_seg_Dataset_22(Dataset):
    """my ver dataset."""

    def __init__(self, ROOT_DIR, transform=None, img_size=(32, 32), random_slice=False):
        """
        Args:
            ROOT_DIR (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # if img_size==(256,256):
        self.image_size = img_size
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             # 下面的img_size是传进来的参数
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))
             ]
        ) if not transform else transform
        self.mask_transform = transforms.Compose(
            [transforms.ToPILImage(),
             # 下面的img_size是传进来的参数
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             ]
        ) if not transform else transform
        self.filenames = os.listdir(ROOT_DIR)  # 返回ROOT_DIR下 所有子目录的名字 的列表
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        if ".ipynb_checkpoints" in self.filenames:
            self.filenames.remove(".ipynb_checkpoints")
        self.ROOT_DIR = ROOT_DIR
        self.random_slice = random_slice

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # print(repr(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()

        save_dir = os.path.join(self.ROOT_DIR, self.filenames[idx])
        img_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "image.jpg")
        unhealthy_mask_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "unhealthy_mask.jpg")
        mask_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "mask.png")

        image = cv2.imread(img_name)
        unhealthy_mask = cv2.imread(unhealthy_mask_name)
        mask = cv2.imread(mask_name)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        unhealthy_mask = cv2.cvtColor(unhealthy_mask, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # 转换为灰度图

        # threshold_value = 128
        # max_value = 255
        # _, mask = cv2.threshold(mask, threshold_value, max_value, cv2.THRESH_BINARY)

        # image = cv2.equalizeHist(image)  # 直方图均衡化

        if self.transform:
            # (256,192)->(256,256)
            # image = np.array(image)
            image = self.transform(image)
            unhealthy_mask = self.mask_transform(unhealthy_mask)
            mask = self.mask_transform(mask)

        sample = {'image': image,
                  'mask': mask,
                  'unhealthy_mask': unhealthy_mask,
                  "filenames": self.filenames[idx],
                  "save_dir": save_dir
                  }
        return sample


class MRIDataset(Dataset):
    """Healthy MRI dataset."""

    def __init__(self, ROOT_DIR, transform=None, img_size=(32, 32), random_slice=False):
        """
        Args:
            ROOT_DIR (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        '''对于预处理，
        我们应用 ±3° 的随机旋转和 0.02×width 和 0.09×height 的随机平移，
        然后中心裁剪 235，调整为 256×256。'''
        # if img_size==(256,256):
        self.image_size = img_size
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomAffine(3, translate=(0.02, 0.09)),
             # transforms.CenterCrop(235),
             # 下面的img_size是传进来的参数
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             # transforms.CenterCrop(256),
             transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))
             ]
        ) if not transform else transform

        self.filenames = os.listdir(ROOT_DIR)  # 返回ROOT_DIR下 所有子目录的名字 的列表
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        if ".ipynb_checkpoints" in self.filenames:
            self.filenames.remove(".ipynb_checkpoints")
        self.ROOT_DIR = ROOT_DIR
        self.random_slice = random_slice

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # print(repr(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if os.path.exists(os.path.join(self.ROOT_DIR, self.filenames[idx], f"{self.filenames[idx]}.npy")):
            # 有npy文件就读。没有npy文件才会读.nii.gz，并且保存为npy
            image = np.load(os.path.join(self.ROOT_DIR, self.filenames[idx], f"{self.filenames[idx]}.npy"))
            pass
        elif os.path.exists(os.path.join(
                self.ROOT_DIR, self.filenames[idx], f"sub-{self.filenames[idx]}_ses-NFB3_T1w.nii.gz"
        )):
            img_name = os.path.join(
                self.ROOT_DIR, self.filenames[idx], f"sub-{self.filenames[idx]}_ses-NFB3_T1w.nii.gz"
            )
            # random between 40 and 130
            # print(nib.load(img_name).slicer[:,90:91,:].dataobj.shape)
            img = nib.load(img_name)
            image = img.get_fdata()

            image_mean = np.mean(image)
            image_std = np.std(image)
            img_range = (image_mean - 1 * image_std, image_mean + 2 * image_std)
            image = np.clip(image, img_range[0], img_range[1])
            image = image / (img_range[1] - img_range[0])
            np.save(
                os.path.join(self.ROOT_DIR, self.filenames[idx], f"{self.filenames[idx]}.npy"), image.astype(
                    np.float32
                )
            )
        else:
            img_name = os.path.join(self.ROOT_DIR, self.filenames[idx])
            image = cv2.imread(img_name)
            # if image is None:
            #     image = cv2.imread('./vertebrae/Train/healthy_1.jpg')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
            # image = cv2.equalizeHist(image)  # 直方图均衡化

        if self.random_slice:
            # slice_idx = randint(32, 122)
            slice_idx = randint(40, 100)
            '''我们使用大脑的 2D 256×192 轴向切片，
                        因为从这个视图通常更容易发现异常。
                        然后我们通过随机选择一个整数 i ∈ [40, 100] 来选择切片。'''
            # (256,256,192)->(256,1,192)
            image = image[:, slice_idx:slice_idx + 1, :]
            # (256,1,192)->(256,192)
            image = image.reshape(256, 192).astype(np.float32)
        else:
            pass

        if self.transform:
            # (256,192)->(256,256)
            # image = np.array(image)
            image = self.transform(image)

        sample = {'image': image,
                  "filenames": self.filenames[idx]}
        return sample

class syn_MRIDataset(Dataset):
    """Healthy MRI dataset with synthetic anomaly."""

    def __init__(self, ROOT_DIR, transform=None, img_size=(32, 32), random_slice=False):
        """
        Args:
            ROOT_DIR (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # if img_size==(256,256):
        self.image_size = img_size
        self.transform = transforms.Compose(
            [
             transforms.ToPILImage(),
             # 下面的img_size是传进来的参数
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))
             ]
        ) if not transform else transform
        self.mask_transform = transforms.Compose(
            [transforms.ToPILImage(),
             # 下面的img_size是传进来的参数
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             ]
        ) if not transform else transform
        self.filenames = os.listdir(ROOT_DIR)  # 返回ROOT_DIR下 所有子目录的名字 的列表
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        if ".ipynb_checkpoints" in self.filenames:
            self.filenames.remove(".ipynb_checkpoints")
        self.ROOT_DIR = ROOT_DIR
        self.random_slice = random_slice

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # print(repr(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()

        save_dir = os.path.join(self.ROOT_DIR, self.filenames[idx])
        img_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "image.png")
        mask_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "mask.png")
        unhealthy_mask_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "unhealthy_mask.png")

        image = cv2.imread(img_name)
        mask = cv2.imread(mask_name)
        unhealthy_mask = cv2.imread(unhealthy_mask_name)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        unhealthy_mask = cv2.cvtColor(unhealthy_mask, cv2.COLOR_BGR2GRAY)  # 转换为灰度图

        # 将 mask 中非零部分的值设置为255
        mask[mask > 0] = 255
        # 将 unhealthy_mask 中非零部分的值设置为255
        unhealthy_mask[unhealthy_mask > 0] = 255
        # image = cv2.equalizeHist(image)  # 直方图均衡化

        # threshold_value = 128
        # max_value = 255
        # _, mask = cv2.threshold(mask, threshold_value, max_value, cv2.THRESH_BINARY)

        input, msk, box_unhealthy_mask = get_IPM_FPM_image(image, "Train", self.filenames[idx])
        input = input.astype(np.uint8)
        # # 查看 input 数组中的最小值和最大值
        # min_value_input = np.min(input)
        # max_value_input = np.max(input)
        #
        # print("input 数组最小值:", min_value_input)
        # print("input 数组最大值:", max_value_input)
        #
        # # 查看 image 数组中的最小值和最大值
        # min_value_image = np.min(image)
        # max_value_image = np.max(image)
        #
        # print("image 数组最小值:", min_value_image)
        # print("image 数组最大值:", max_value_image)


        if self.transform:
            # (256,192)->(256,256)
            # image = np.array(image)
            image = self.transform(image)
            mask = self.mask_transform(mask)
            unhealthy_mask = self.mask_transform(unhealthy_mask)
            input = self.transform(input)
            msk = self.mask_transform(msk)
            box_unhealthy_mask = self.mask_transform(box_unhealthy_mask)

        # min_value = torch.min(input)
        # max_value = torch.max(input)
        #
        # print("input最小值:", min_value.item())
        # print("input最大值:", max_value.item())
        #
        # min_value = torch.min(image)
        # max_value = torch.max(image)
        #
        # print("image最小值:", min_value.item())
        # print("image最大值:", max_value.item())
        sample = {'image': image,
                  'input': input,
                  'box_healthy_mask': msk,
                  'box_unhealthy_mask': box_unhealthy_mask,
                  'mask': mask,
                  'unhealthy_mask': unhealthy_mask,
                  "id": self.filenames[idx],
                  "save_dir": save_dir
                  }
        return sample
