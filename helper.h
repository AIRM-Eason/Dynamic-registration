#ifndef DYNAMIC_REGISTRATION_HELPER_H
#define DYNAMIC_REGISTRATION_HELPER_H

#include <iostream>
#include <tuple>
#include <array>
#include <cmath>
#include <algorithm>
#include <iterator>

#include <librealsense2/rs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/cvstd.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

using std::string;
using std::vector;
using std::pair;
using std::cout;
using std::endl;
using std::ostream;

// Estimate the rotation matrix from gyroscope measurements.
class RotationEstimator {
private:
    Eigen::Vector3d rotation = {0, 0, 0};
    Eigen::Vector3d angular_velocity = {0, 0, 0};

    bool first = true;
    double last_ts = 0;
    double delta_ts = 0;

public:
    void reset() {
        first = true;
        rotation = {0, 0, 0};
    }

    void print_rotation() {
        std::cout << "Rotation: " <<
                  rotation[0] << ", " <<
                  rotation[1] << ", " <<
                  rotation[2] << std::endl;
    }

    void process(rs2_vector gyro_data, double ts);

    Eigen::Matrix3d get_rotation_mat() const;

};

struct Point2d {
    double x, y;

    // Calculate the distance to the other Point2d using L2 nrom.
    double l2(const Point2d &p) const {
        return std::sqrt(std::pow(x - p.x, 2) + std::pow(y - p.y, 2));
    }
};

struct Point3d {
    double x, y, z;

    // Calculate the distance to the other Point3d using L2 nrom.
    double l2(const Point3d &p) const {
        return std::sqrt(std::pow(x - p.x, 2) + std::pow(y - p.y, 2) + std::pow(z - p.z, 2));
    }
};

/*
 * Create View class as a data structure to store all the data for each frame
 * @rotation Rotation matrix from last view to current view
 * @img Grayscale image for feature detection
 * @img_coords 2d image coordinates of points in the point cloud
 * @cam_coords 3d world coordinates of points in the point cloud
 * @rgb_cloud pcl rgb point cloud in current view that can be transformed to the next view
 */
struct View {
    Eigen::Matrix3d rotation;
    cv::Mat img;
    vector<Point2d> img_coords;
    vector<Point3d> cam_coords;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud;

    View(Eigen::Matrix3d rotation, cv::Mat img, vector<Point2d> img_coords, vector<Point3d> cam_coords,
         pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud) : rotation(std::move(rotation)), img(std::move(img)),
                                                             img_coords(std::move(img_coords)),
                                                             cam_coords(std::move(cam_coords)),
                                                             rgb_cloud(std::move(rgb_cloud)) {}
};

cv::Mat frame_to_mat(const rs2::video_frame &rgb_frame);

void get_rgb(const uint8_t *raw_rgb, int stride, double x, double y, uint8_t *rgb);

pcl::PointCloud<pcl::PointXYZRGB>::Ptr
get_point_cloud(const rs2::depth_frame &depth, const rs2::video_frame &rgb, vector<Point2d> &img_coords,
                vector<Point3d> &cam_coords);

vector<View> get_views(const string &fn, const vector<double> &timestamps);

Eigen::Matrix4d
estimate(const View &view1, const View &view2, int num_features = 1000, float ratio_thresh = 0.55f);

std::pair<Eigen::Matrix4d, vector<double>>
affine_estimation(vector<pair<Point3d, Point3d>> correspondences, Eigen::Matrix3d rotation);

Eigen::Matrix4d
robust_fitting(vector<pair<Point3d, Point3d>> correspondences, Eigen::Matrix3d R, vector<int> &removed_idx);

void combine_pc(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_ref, pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc,
                Eigen::Matrix4d affine);

void visualize_cloud(pcl::visualization::PCLVisualizer::Ptr visualizer,
                     pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr rgb_cloud);

#endif //DYNAMIC_REGISTRATION_HELPER_H
