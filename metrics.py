# -*- coding:utf-8 -*-

import numpy as np
import scipy.ndimage as ndimage
from skimage import morphology
import nibabel as nib


# Dice
def Dice(inputs, targets):
    smooth = 1e-6
    inputs = inputs.flatten()
    targets = targets.flatten()
    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return dice


def Contain(inputs, targets):
    inputs = inputs.flatten()
    targets = targets.flatten()
    num = np.count_nonzero(targets)
    intersection = (inputs * targets).sum()
    contain = intersection / num
    return contain

import GeodisTK
# Hausdorff and ASSD evaluation
def get_edge_points(img):
    """
    get edge points of a binary segmentation result
    """
    dim = len(img.shape)
    if dim == 2:
        strt = ndimage.generate_binary_structure(2, 1)
    else:
        strt = ndimage.generate_binary_structure(3, 1)  # 三维结构元素，与中心点相�?个像素点的都是邻�?
    ero = ndimage.morphology.binary_erosion(img, strt)
    edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
    return edge


def binary_assd(s, g, data_mode, spacing=None):
    """
    get the average symetric surface distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """

    s = convertion(s, mode=data_mode)
    s = s[:g.shape[0], :g.shape[1], :g.shape[2]]
    s, g = s.astype(np.uint8), g.astype(np.uint8)
    s_edge = morphology.skeletonize(s)
    #s_edge = np.array(s, dtype=np.bool)
    #g_edge = morphology.skeletonize(g)
    g_edge = g
    image_dim = len(s.shape)
    assert image_dim == len(g.shape)
    if spacing is None:
        spacing = [1.0] * image_dim
    else:
        assert image_dim == len(spacing)
    img = np.zeros_like(s)
    if image_dim == 2:
        # compute the closest distance of each voxel to the edge voxels
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif image_dim == 3:
        # compute the closest distance of each voxel to the edge voxels
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    ns = s_edge.sum()
    ng = g_edge.sum()
    s_dis_g_edge = s_dis * g_edge
    g_dis_s_edge = g_dis * s_edge
    assd = (s_dis_g_edge.sum() + g_dis_s_edge.sum()) / (ns + ng)
    #assd = (s_dis_g_edge.sum() + g_dis_s_edge.sum()) / 2
    return assd, s_edge, g_edge, s_dis, g_dis


def binary_asd(s, g, data_mode, spacing=None):
    """
    get the average symetric surface distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """

    s = convertion(s, mode=data_mode)
    s = s[:g.shape[0], :g.shape[1], :g.shape[2]]
    s, g = s.astype(np.uint8), g.astype(np.uint8)
    s_edge = morphology.skeletonize(s)
    #s_edge = np.array(s, dtype=np.bool)
    #g_edge = morphology.skeletonize(g)
    g_edge = g
    image_dim = len(s.shape)
    assert image_dim == len(g.shape)
    if spacing is None:
        spacing = [1.0] * image_dim
    else:
        assert image_dim == len(spacing)
    img = np.zeros_like(s)
    if image_dim == 2:
        # compute the closest distance of each voxel to the edge voxels
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif image_dim == 3:
        # compute the closest distance of each voxel to the edge voxels
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        #g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    #ns = s_edge.sum()
    ng = g_edge.sum()
    s_dis_g_edge = s_dis * g_edge
    #g_dis_s_edge = g_dis * s_edge
    assd = s_dis_g_edge.sum() / (ng)
    #assd = (s_dis_g_edge.sum() + g_dis_s_edge.sum()) / 2
    return assd, s_edge, g_edge, s_dis

def binary_ov(s_edge, g_edge, s_dis, g_dis):
    dist1 = 1.0
    dist2 = 1.5
    s_dis_g_edge = s_dis[g_edge > 0]
    g_dis_s_edge = g_dis[s_edge > 0]
    TPM1 = np.sum(s_dis_g_edge <= dist1)
    FPM1 = np.sum(s_dis_g_edge > dist1)
    TPMR1 = TPM1 / (TPM1 + FPM1)
    TPR1 = np.sum(g_dis_s_edge <= dist1)
    FNR1 = np.sum(g_dis_s_edge > dist1)
    TPRR1 = TPR1 / (TPR1 + FNR1)
    OV1 = (TPM1 + TPR1) / (TPM1 + FPM1 + TPR1 + FNR1)

    TPM2 = np.sum(s_dis_g_edge <= dist2)
    FPM2 = np.sum(s_dis_g_edge > dist2)
    TPMR2 = TPM2 / (TPM2 + FPM2)
    TPR2 = np.sum(g_dis_s_edge <= dist2)
    FNR2 = np.sum(g_dis_s_edge > dist2)
    TPRR2 = TPR2 / (TPR2 + FNR2)
    OV2 = (TPM2 + TPR2) / (TPM2 + FPM2 + TPR2 + FNR2)
    return OV1, TPMR1, TPRR1, OV2, TPMR2, TPRR2

def binary_aov(s_edge, g_edge, s_dis):
    dist1 = 1.0
    dist2 = 1.5
    s_dis_g_edge = s_dis[g_edge > 0]
    TPM1 = np.sum(s_dis_g_edge <= dist1)
    FPM1 = np.sum(s_dis_g_edge > dist1)
    OV1= TPM1 / (TPM1 + FPM1)

    TPM2 = np.sum(s_dis_g_edge <= dist2)
    FPM2 = np.sum(s_dis_g_edge > dist2)
    OV2 = TPM2 / (TPM2 + FPM2)

    return OV1, OV2


def binary_hausdorff95(s_edge, g_edge, s_dis, g_dis):
    dist_list1 = s_dis[g_edge > 0]
    dist_list1 = sorted(dist_list1)
    dist1 = dist_list1[int(len(dist_list1) * 0.95)]
    dist_list2 = g_dis[s_edge > 0]
    dist_list2 = sorted(dist_list2)
    if len(dist_list2)==0:
        dist2 = 0
    else:
        dist2 = dist_list2[int(len(dist_list2) * 0.95)]
    return max(dist1, dist2)

