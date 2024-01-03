#14+4个指标
import argparse
import csv
import glob
import json
import os.path
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d

parser = argparse.ArgumentParser(description='Calculate feature for cell from Hovernet')
parser.add_argument('--patch_dir', type=str, default='/home/duxianglong/project/subproject/dxl/hover_net-master/datasets/thymus_all/slide_all', help='directory for jpeg')
parser.add_argument('--json_dir', type=str, default='/home/duxianglong/project/subproject/dxl/hover_net-master/datasets/thymus_all/result_all', help='directory for json')
parser.add_argument('--result_dir', type=str, default='/home/duxianglong/project/subproject/dxl/hover_net-master/datasets/thymus_all/cell_analysis/feature', help='directory for result to save')
parser.add_argument('--extension', type=str, default='.jpeg', help='path format')
args = parser.parse_args()
###特征文件对应列
headline=['cellType','area','bbox_area','major_axis_length','minor_axis_length','eccentricity','perimeter','circularity',
          'elongation','extent','solidity','curve_mean','curve_max','curve_min','curve_std',
          'intensity_r_mean','intensity_r_std','intensity_r_max','intensity_r_min',
          'intensity_g_mean','intensity_g_std','intensity_g_max','intensity_g_min',
          'intensity_b_mean','intensity_b_std','intensity_b_max','intensity_b_min']

# 展示要计算的细胞
def Show_cell(rand_centroid,rand_bbox,rand_contour,image):
    overlay = image.copy()
    overlay = cv2.drawContours(overlay.astype('uint8'), [np.array(rand_contour)], -1, (255,255,0), 1)
    overlay = cv2.circle(overlay.astype('uint8'),(np.round(rand_centroid[0]).astype('int'), np.round(rand_centroid[1]).astype('int')), 3, (0,255,0), -1)
    overlay = cv2.rectangle(overlay.astype('uint8'), (rand_bbox[0][1], rand_bbox[0][0]), (rand_bbox[1][1], rand_bbox[1][0]), (255,0,0), 1)
    plt.imshow(overlay)
    plt.axis('off')
    plt.title('Overlay', fontsize=25)
    plt.show()
def Cell_area(inst_contour):
    area = cv2.contourArea(np.array(inst_contour))
    return round(area,4)

def Cell_bbox_area(inst_bbox): ###左上顶点和右下顶点坐标
    return round((inst_bbox[1][1]-inst_bbox[0][1])*(inst_bbox[1][0]-inst_bbox[0][0]),4)

###丐版
def Cell_major_axis_length_1(inst_centroid,inst_contour):
    points = np.array(inst_contour)
    distance = np.linalg.norm(points - inst_centroid, axis=1)
    max_distance = round(np.max(distance),2)*2
    min_distance = round(np.min(distance), 2)*2
    return round(max_distance),round(min_distance)
###是否验证正确？1和 3返回结果相近

def Cell_major_axis_length_2(inst_centroid,inst_contour):
    x = [point[0] for point in inst_contour]
    y = [point[1] for point in inst_contour]
    # 将坐标转化为 NumPy 数组
    coords = np.array([x, y])
    # 计算协方差矩阵 表明不同元素之间的协方差(线性关系），对角线为方差
    cov_matrix = np.cov(coords)
    # 计算协方差矩阵的特征值和特征向量
    eigvals, eigvecs = np.linalg.eig(cov_matrix)
    # 计算主轴和次轴长度
    major_axis_length = np.sqrt(max(eigvals)) * 2
    minor_axis_length = np.sqrt(min(eigvals)) * 2
    return round(major_axis_length,4),round(minor_axis_length,4)

def Cell_major_axis_length_3(inst_centroid,inst_contour):
    # 拟合椭圆
    inst_contour = inst_contour
    if len(inst_contour) > 5:
        new_inst_contour = np.array(inst_contour)
    else:
        return -1,-1
        # new_inst_contour=[]
        # for i in range(len(inst_contour)-1):
        #     p1 = inst_contour[i]
        #     p2 = inst_contour[(i+1) % len(inst_contour)]
        #     mid_point = [(p1[0]+p2[0])//2, (p1[1]+p2[1])//2]
        #     new_inst_contour.extend([p1,mid_point])
        # new_inst_contour = np.array(new_inst_contour)
    ellipse = cv2.fitEllipse(new_inst_contour)
    return max(ellipse[1]),min(ellipse[1])

def Cell_eccentricity(major_axis_length,minor_axis_length):
    return round(np.sqrt(1 - (minor_axis_length / major_axis_length)**2),4)

# 计算两个点之间的距离
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def Cell_perimeter(inst_contour):
    # 将坐标对转换为NumPy数组
    points = np.array(inst_contour)
    # 计算相邻点之间的距离
    distances = [calculate_distance(points[i], points[i + 1]) for i in range(len(points) - 1)]
    # 加上首尾两个点之间的距离
    distances.append(calculate_distance(points[-1], points[0]))
    # 计算边界的周长
    length = sum(distances)
    return round(length,4)

def Cell_circularity(area,perimeter):
    return round((4 * math.pi * area) / (perimeter * perimeter),4)

def Cell_elongation(major_axis_length,minor_axis_length):
    # 长宽比
    return round(major_axis_length/minor_axis_length,4)

def Cell_extent(area,bbox_area):
    # 计算凸包面积 hull_area = cv2.contourArea(cv2.convexHull(np.array(contour)))
    return round(area/bbox_area, 4)

def Cell_solidity(area,inst_contour):
    hull = ConvexHull(inst_contour)
    T_area = hull.volume
    return round(area/T_area, 4)

def curvature(inst_contour):
    inst_contour.append(inst_contour[0])
    a = np.array(inst_contour)
    x = a[:, 0]
    y = a[:, 1]
    t = np.arange(x.shape[0])
    fx = interp1d(t, x, kind='cubic')
    fy = interp1d(t, y, kind='cubic')
    tt = np.linspace(t[0], t[-1], 1000)
    xx = fx(tt)
    yy = fy(tt)
    dx = np.gradient(xx)
    dy = np.gradient(yy)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvatures = np.abs((dx * ddy - dy * ddx))/ np.power(dx**2 + dy**2, 1.5)
    return round(np.mean(curvatures),4), round(np.max(curvatures),4),round(np.min(curvatures),4), round(np.var(curvatures),4)

def Cell_instensity(image,inst_contour):
    mask = np.zeros((256,256),dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(inst_contour)], 255)
    red_channel = image[:, :, 2]  ## 提取红色通道 红色通道索引为2
    green_channel = image[:, :, 1]  ## 提取绿色通道 绿色通道索引为1
    blue_channel = image[:, :, 0]  ## 提取蓝色通道 蓝色通道索引为0
    region_read = cv2.bitwise_and(red_channel,red_channel,mask=mask)
    region_green = cv2.bitwise_and(green_channel,green_channel,mask=mask)
    region_blue = cv2.bitwise_and(blue_channel,blue_channel,mask=mask)
    intensity_r = [np.mean(region_read[mask==255]),np.std(region_read[mask==255]),np.max(region_read[mask==255]),np.min(region_read[mask==255])] ##均值，方差，最大值，最小值
    intensity_g = [np.mean(region_green[mask==255]), np.std(region_green[mask==255]), np.max(region_green[mask==255]), np.min(region_green[mask==255])]  ##均值，方差，最大值，最小值
    intensity_b = [np.mean(region_blue[mask==255]), np.std(region_blue[mask==255]), np.max(region_blue[mask==255]), np.min(region_blue[mask==255])]  ##均值，方差，最大值，最小值
    intensity_r = [round(_,4) for _ in intensity_r]
    intensity_g = [round(_, 4) for _ in intensity_g]
    intensity_b = [round(_, 4) for _ in intensity_b]
    return intensity_r,intensity_g,intensity_b

def CalculateFeature(jpg_path,json_path,saveFile):

    with open(json_path) as jsonfile:
        data = json.load(jsonfile)
    mag_info=data['mag']
    nuc_info = data['nuc']
    # mag表示图像中强度或幅度信息（对比度） nuclear: 细胞核相关信息 ：边界框，质心，轮廓坐标列表，类型概率，类型

    patchData=[]

    for process_i,inst in enumerate(nuc_info):
        cellData=[]
        inst_info = nuc_info[inst]
        inst_centroid = inst_info['centroid']  ###中心点坐标
        inst_contour = inst_info['contour']   ###轮廓坐标点
        inst_bbox = inst_info['bbox']   ###BBox 外切矩形
        inst_type = inst_info['type']   ###细胞类别0-5

        ###展示细胞
        image = cv2.imread(jpg_path)
        # Show_cell(inst_centroid,inst_bbox,inst_contour,image)

        ###特征计算14+4
        area = Cell_area(inst_contour) ### 使用细胞的轮廓坐标计算细胞的像素面积
        bbox_area = Cell_bbox_area(inst_bbox) ### 使用外切矩形坐标计算像素面积
        major_axis_length,minor_axis_length = Cell_major_axis_length_3(inst_centroid,inst_contour) ###主轴和次轴长度
        if minor_axis_length == 0 or major_axis_length==-1:
            continue
        eccentricity = Cell_eccentricity(major_axis_length,minor_axis_length) ###偏心率  需要预先计算主轴和次轴长度
        perimeter = Cell_perimeter(inst_contour) ###细胞周长
        circularity = Cell_circularity(area,perimeter) ###细胞圆度
        elongation = Cell_elongation(major_axis_length,minor_axis_length) ###延伸率
        extent = Cell_extent(area,bbox_area) ###空间占比 预先计算area 和 bbox_area
        solidity = Cell_solidity(area,inst_contour) ###实心度
        curve_mean, curve_max, curve_min, curve_std = curvature(inst_contour) ###曲率均值，最大，最小，方差
        # curve_mean, curve_max, curve_min, curve_std=1,1,1,1
        intensity_r,intensity_g,intensity_b = Cell_instensity(image,inst_contour) ###色彩强度 Instensity 按序返回 均值，方差，最大值，最小值
        # intensity_r, intensity_g, intensity_b=[1,1,1,1],[1,1,1,1],[1,1,1,1]
        ###特征保存
        cellData.extend([str(inst_type),str(area),str(bbox_area),str(major_axis_length),str(minor_axis_length),str(eccentricity),str(perimeter),str(circularity),
                          str(elongation),str(extent),str(solidity),str(curve_mean),str(curve_max),str(curve_min),str(curve_std)])
        intensity_r = [str(_) for _ in intensity_r]
        intensity_g = [str(_) for _ in intensity_g]
        intensity_b = [str(_) for _ in intensity_b]
        cellData.extend(intensity_r)
        cellData.extend(intensity_g)
        cellData.extend(intensity_b)
        patchData.append(cellData)
    with open(saveFile,'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(patchData)

def main():
    jsonAll = glob.glob(args.json_dir+"/*/json")
    for process_i,json_item in enumerate(jsonAll):
        slideName = json_item.split("/")[-2]
        patchJson = glob.glob(json_item+"/*.json")
        print(f"{process_i} / {len(jsonAll)}    {slideName},    patch total:{len(patchJson)}")
        ###创建特征保存文件
        saveFile = os.path.join(args.result_dir, slideName + ".csv")
        with open(saveFile, 'w', newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headline)
        for patch_item in tqdm(patchJson):
            patchName = patch_item.split("/")[-1].rstrip(".json")+args.extension
            pathJpg = os.path.join(args.patch_dir,slideName,patchName)
            if not os.path.exists(pathJpg):
                print(f"{pathJpg} corresponging image not exsits!!!")
                return
            CalculateFeature(pathJpg,patch_item,saveFile)



if __name__ == '__main__':
    main()
