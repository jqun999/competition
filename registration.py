import numpy as np
import cv2
import SimpleITK as sitk
from scipy.interpolate import griddata
from skimage.transform import AffineTransform, warp
import matplotlib.pyplot as plt

# 1. 读取医学造影图像
def load_medical_image(image_path):
    image = sitk.ReadImage(image_path)
    return image

# 2. 处理光纤传感数据（假设光纤数据为三维曲线数据：x, y, z）
def process_fiber_sensor_data(sensor_data):
    # 假设传感数据是一个N x 3矩阵（x, y, z坐标）
    # 可以进行滤波或平滑处理
    # 例如使用滑动平均滤波：
    smoothed_data = np.convolve(sensor_data[:, 2], np.ones(5)/5, mode='same')
    sensor_data[:, 2] = smoothed_data
    return sensor_data

# 3. 医学图像预处理（去噪，增强对比度）
def preprocess_medical_image(image):
    image = sitk.GetArrayFromImage(image)
    # 简单的图像去噪处理
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image

# 4. 图像配准（使用刚性配准方法）
def register_images(fiber_image, medical_image):
    # 使用基于互信息的配准方法进行刚性配准
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fiber_image)
    elastixImageFilter.SetMovingImage(medical_image)
    elastixImageFilter.Execute()
    registered_image = elastixImageFilter.GetResultImage()
    return registered_image

# 5. 数据融合：将光纤传感数据映射到医学图像中
def fuse_data_to_image(sensor_data, medical_image, affine_transform):
    # 将光纤传感数据通过仿射变换映射到图像坐标系
    transformed_data = affine_transform(sensor_data[:, 0], sensor_data[:, 1])

    # 在医学图像上绘制光纤传感数据
    fig, ax = plt.subplots()
    ax.imshow(medical_image, cmap='gray')
    ax.plot(transformed_data[:, 0], transformed_data[:, 1], color='r')
    plt.show()

# 6. 导丝形状重构：补偿误差
def reconstruct_shape(sensor_data, medical_image):
    # 假设我们有经过配准的图像，使用插值重建导丝形状
    # 使用网格数据插值来重建形状
    grid_x, grid_y = np.mgrid[0:medical_image.shape[0], 0:medical_image.shape[1]]
    reconstructed_shape = griddata(sensor_data[:, :2], sensor_data[:, 2], (grid_x, grid_y), method='cubic')
    return reconstructed_shape

# 7. 主函数（执行光纤数据与医学图像配准与融合）
def main(sensor_data_path, medical_image_path):
    # 读取传感数据与医学图像
    sensor_data = np.loadtxt(sensor_data_path)  # 假设传感数据是一个文件
    medical_image = load_medical_image(medical_image_path)

    # 处理光纤传感数据
    sensor_data = process_fiber_sensor_data(sensor_data)

    # 预处理医学图像
    processed_medical_image = preprocess_medical_image(medical_image)

    # 假设已经有配准好的图像
    # 进行图像配准
    fiber_image = processed_medical_image  # 假设这里的fiber_image为医学图像中的光纤传感影像
    registered_image = register_images(fiber_image, processed_medical_image)

    # 数据融合：将光纤数据与医学图像对齐并可视化
    affine_transform = AffineTransform(scale=(1, 1), rotation=0, translation=(0, 0))  # 示例仿射变换
    fuse_data_to_image(sensor_data, registered_image, affine_transform)

    # 导丝形状重构
    reconstructed_shape = reconstruct_shape(sensor_data, processed_medical_image)

    # 显示重构的导丝形状
    plt.imshow(reconstructed_shape, cmap='jet')
    plt.colorbar()
    plt.show()

# 调用主函数
main('sensor_data.txt', 'medical_image.mhd')
