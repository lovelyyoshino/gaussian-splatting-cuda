#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <point_cloud.cuh>
#include <image.cuh>
#include <camera_info.cuh>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


using PointCloudXYZRGB = pcl::PointCloud<pcl::PointXYZRGB>;

std::unordered_map<int, std::pair<CAMERA_MODEL, uint32_t>> write_camera_model_ids = {
    {0, {CAMERA_MODEL::SIMPLE_PINHOLE, 3}},
    {1, {CAMERA_MODEL::PINHOLE, 4}},
    {2, {CAMERA_MODEL::SIMPLE_RADIAL, 4}},
    {3, {CAMERA_MODEL::RADIAL, 5}},
    {4, {CAMERA_MODEL::OPENCV, 8}},
    {5, {CAMERA_MODEL::OPENCV_FISHEYE, 8}},
    {6, {CAMERA_MODEL::FULL_OPENCV, 12}},
    {7, {CAMERA_MODEL::FOV, 5}},
    {8, {CAMERA_MODEL::SIMPLE_RADIAL_FISHEYE, 4}},
    {9, {CAMERA_MODEL::RADIAL_FISHEYE, 5}},
    {10, {CAMERA_MODEL::THIN_PRISM_FISHEYE, 12}},
    {11, {CAMERA_MODEL::UNDEFINED, -1}}};

struct Track {
    uint32_t _image_ID;
    uint32_t _max_num_2D_points;
};

// 保存图像数据的函数
void save_images(const std::filesystem::path& file_path, const std::vector<std::pair<cv::Mat,int>>& images, const std::vector<Eigen::Matrix4d>& poses) {
    if (images.size() != poses.size()) {
        throw std::invalid_argument("Images and poses must have the same size.");
    }

    std::ofstream output_stream(file_path / "images.bin", std::ios::binary);
    if (!output_stream) {
        throw std::runtime_error("Unable to open file for writing: " + (file_path / "images.bin").string());
    }

    uint64_t image_count = images.size();
    output_stream.write(reinterpret_cast<const char*>(&image_count), sizeof(image_count));
    struct ImagePoint { // we dont need this later
        double _x;
        double _y;
        uint64_t _point_id;
    };
    for (size_t i = 0; i < image_count; ++i) {
        std::pair<cv::Mat,int> img = images[i];
        const auto& pose = poses[i];

        // 从Matrix4d中提取旋转和平移
        Eigen::Quaterniond q(pose.block<3, 3>(0, 0)); // 提取旋转部分
        Eigen::Vector3d t = pose.block<3, 1>(0, 3); // 提取平移部分

        // 写入相机 ID（可以使用索引作为 ID）
        uint32_t image_id = static_cast<uint32_t>(i); // 使用索引作为相机 ID
        output_stream.write(reinterpret_cast<const char*>(&image_id), sizeof(image_id));

        // 写入四元数
        output_stream.write(reinterpret_cast<const char*>(&q.coeffs()), sizeof(double) * 4); // 四元数的四个系数
        // 写入平移向量
        output_stream.write(reinterpret_cast<const char*>(&t), sizeof(Eigen::Vector3d));

        // 写入相机 ID（可以使用索引作为 ID）
        uint32_t camera_id = static_cast<uint32_t>(img.second); // 使用索引作为相机 ID
        output_stream.write(reinterpret_cast<const char*>(&camera_id), sizeof(camera_id));

        // 写入图像名称（使用索引作为名称）
        std::string image_name = "image_" + std::to_string(i) + ".png"; // 假设为 PNG 文件
        output_stream.write(image_name.c_str(), image_name.size() + 1); // 包括空字符
# if 0
        // 写入图像数据大小
        uint64_t number_points = img.first.total(); // 获取图像数据的字节大小
        output_stream.write(reinterpret_cast<const char*>(&number_points), sizeof(number_points)); // 写入图像大小
        std::vector<ImagePoint> points(number_points);
        for(int i = 0; i < img.first.rows; i++) {
            for(int j = 0; j < img.first.cols; j++) {
                points[i*img.first.cols+j]._x = j;
                points[i*img.first.cols+j]._y = i;
                points[i*img.first.cols+j]._point_id = i*img.first.cols+j ;
            }
        }
        // 写入所有点数据
        output_stream.write(reinterpret_cast<char*>(points.data()), number_points * sizeof(ImagePoint));
#else
        uint64_t number_points = 0;
        output_stream.write(reinterpret_cast<const char*>(&number_points), sizeof(number_points)); // 写入图像大小
#endif

    }

    output_stream.close();
}

void save_cameras(const std::filesystem::path& file_path, 
                  const std::vector<std::tuple<uint32_t, int, uint64_t, uint64_t, std::vector<double>>>& cameras) {
    std::ofstream output_stream(file_path / "cameras.bin", std::ios::binary);
    if (!output_stream) {
        throw std::runtime_error("Unable to open file for writing: " + (file_path / "cameras.bin").string());
    }

    uint64_t camera_count = cameras.size();
    output_stream.write(reinterpret_cast<const char*>(&camera_count), sizeof(camera_count));

    for (const auto& params : cameras) {
        uint32_t camera_id = std::get<0>(params);
        // 解包参数
        int model_id = std::get<1>(params);
        uint64_t width = std::get<2>(params);
        uint64_t height = std::get<3>(params);
        const auto& camera_parameters = std::get<4>(params);

        // 写入相机 ID
        output_stream.write(reinterpret_cast<const char*>(&camera_id), sizeof(camera_id));
        
        // 写入相机模型
        output_stream.write(reinterpret_cast<const char*>(&model_id), sizeof(model_id));
        
        // 写入图像尺寸
        output_stream.write(reinterpret_cast<const char*>(&width), sizeof(width));
        output_stream.write(reinterpret_cast<const char*>(&height), sizeof(height));
        CAMERA_MODEL camera_model  = std::get<0>(write_camera_model_ids[model_id]);
        uint32_t camera_model_id = std::get<1>(write_camera_model_ids[model_id]);
        output_stream.write(reinterpret_cast<const char*>(&camera_model), sizeof(camera_model));
        output_stream.write(reinterpret_cast<const char*>(&camera_model_id), sizeof(camera_model_id));
        // 写入参数
        size_t params_size = camera_parameters.size();
        output_stream.write(reinterpret_cast<const char*>(camera_parameters.data()), params_size * sizeof(double));
    }
    output_stream.close();
}


void save_point_cloud(const std::filesystem::path& file_path, const PointCloudXYZRGB& cloud, const std::vector<std::vector<Track>>& tracks) {
    std::ofstream output_stream(file_path, std::ios::binary);
    if (!output_stream) {
        throw std::runtime_error("Unable to open file for writing: " + file_path.string());
    }

    // 写入点云的点数量
    uint64_t point3D_count = cloud.size();
    output_stream.write(reinterpret_cast<const char*>(&point3D_count), sizeof(point3D_count));

    for (size_t i = 0; i < point3D_count; ++i) {
        const auto& point = cloud.points[i];

        // 写入点的坐标 (x, y, z)
        double x = static_cast<double>(point.x);
        double y = static_cast<double>(point.y);
        double z = static_cast<double>(point.z);
        output_stream.write(reinterpret_cast<const char*>(&x), sizeof(x));
        output_stream.write(reinterpret_cast<const char*>(&y), sizeof(y));
        output_stream.write(reinterpret_cast<const char*>(&z), sizeof(z));

        // 写入颜色 (r, g, b)
        uint8_t r = point.r;
        uint8_t g = point.g;
        uint8_t b = point.b;
        output_stream.write(reinterpret_cast<const char*>(&r), sizeof(r));
        output_stream.write(reinterpret_cast<const char*>(&g), sizeof(g));
        output_stream.write(reinterpret_cast<const char*>(&b), sizeof(b));

        // 写入被忽略的值（如深度）
        double ignored_value = 0.0; // 这里可以根据实际需要设定
        output_stream.write(reinterpret_cast<const char*>(&ignored_value), sizeof(ignored_value));
# if 0
        uint64_t track_length = tracks[i].size();
        output_stream.write(reinterpret_cast<const char*>(&track_length), sizeof(track_length));
        output_stream.write(reinterpret_cast<const char*>(tracks[i].data()), track_length * sizeof(uint32_t));
#else
        uint64_t track_length = 0;
        output_stream.write(reinterpret_cast<const char*>(&track_length), sizeof(track_length));
#endif
    }

    output_stream.close();
}


// 创建假数据并调用保存函数
int main() {
    // 假设的图像数据和位姿
    std::vector<std::pair<cv::Mat, int>> images;
    std::vector<Eigen::Matrix4d> poses;
    
    for (int i = 0; i < 5; ++i) { // 创建 5 张假图像
        cv::Mat img = cv::Mat::zeros(480, 640, CV_8UC3); // 黑色图像
        images.emplace_back(img, 0); // 使用相机 ID 0
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity(); // 单位矩阵表示无旋转和平移
        poses.push_back(pose);
    }

    // 保存图像
    std::filesystem::path image_path = "./";
    save_images(image_path, images, poses);

    // 假设的相机参数
    std::vector<std::tuple<uint32_t, int, uint64_t, uint64_t, std::vector<double>>> cameras;
    for (int i = 0; i < 5; ++i) {
        uint32_t camera_id = static_cast<uint32_t>(i);
        int model_id = 0; // 使用简单针孔模型
        uint64_t width = 640;
        uint64_t height = 480;
        std::vector<double> camera_parameters = {320.0, 240.0, 600.0}; // fx, fy, cx
        cameras.emplace_back(camera_id, model_id, width, height, camera_parameters);
    }

    // 保存相机参数
    std::filesystem::path camera_path = "./";
    save_cameras(camera_path, cameras);

    // 创建假点云数据
    PointCloudXYZRGB cloud;
    for (int i = 0; i < 1000; ++i) {
        pcl::PointXYZRGB point;
        point.x = static_cast<float>(rand() % 100) / 10.0f; // 随机生成点坐标
        point.y = static_cast<float>(rand() % 100) / 10.0f;
        point.z = static_cast<float>(rand() % 100) / 10.0f;
        point.r = static_cast<uint8_t>(rand() % 256); // 随机颜色
        point.g = static_cast<uint8_t>(rand() % 256);
        point.b = static_cast<uint8_t>(rand() % 256);
        cloud.points.push_back(point);
    }
    cloud.width = cloud.size();
    cloud.height = 1; // 点云数据是未组织的

    // 保存点云数据
    std::filesystem::path point_cloud_path = "./point_cloud.bin";
    save_point_cloud(point_cloud_path, cloud, {});

    std::cout << "Data saved successfully!" << std::endl;

    return 0;
}
