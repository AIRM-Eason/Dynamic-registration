#include "helper.h"

ostream &operator<<(ostream &os, Point2d p) {
    os << "[" << p.x << ", " <<
       p.y << "]";
    return os;
}

ostream &operator<<(ostream &os, Point3d p) {
    os << "[" << p.x << ", " <<
       p.y << ", " <<
       p.z << "]";
    return os;
}

void RotationEstimator::process(rs2_vector gyro_data, double ts) {
    // For the initial frame, only record angular velocity v(0)
    // For the frame at timestamp t, add delta * v(t-1) to the rotation angles
    if (first) {
        last_ts = ts;
        first = false;
    }
    delta_ts = (ts - last_ts) / 1000.0;
    rotation -= angular_velocity * delta_ts; // -=: rotation t1->t0 is negative of rotation t0->t1
    // In this example, the camera is always attached to the ground, rotation only happens in y-axis.
    // Y-axis in pcl library is inverted. -0.0025 is used to compensate sensor bias of gyroscope.
    angular_velocity = {0, -(gyro_data.y - 0.0025), 0};
    last_ts = ts;
}

// Get rotation matrix from Euler angles.
Eigen::Matrix3d RotationEstimator::get_rotation_mat() const {
    Eigen::Matrix3d rx, ry, rz;
    rx << 1, 0, 0,
            0, cos(rotation[0]), -sin(rotation[0]),
            0, sin(rotation[0]), cos(rotation[0]);
    ry << cos(rotation[1]), 0, sin(rotation[1]),
            0, 1, 0,
            -sin(rotation[1]), 0, cos(rotation[1]);
    rz << cos(rotation[2]), -sin(rotation[2]), 0,
            sin(rotation[2]), cos(rotation[2]), 0,
            0, 0, 1;
    return rx * ry * rz;
}

// Convert rgb rs frame to grayscale OpenCV matrix that will be used for feature detection.
cv::Mat frame_to_mat(const rs2::video_frame &rgb_frame) {
    cv::Mat rgb_img, gray_img;
    cv::Size sz(rgb_frame.get_width(), rgb_frame.get_height());
    void *data = const_cast<void *>(rgb_frame.get_data());
    rgb_img = cv::Mat(sz, CV_8UC3, data);
    cv::cvtColor(rgb_img, gray_img, cv::COLOR_RGB2GRAY);
    return gray_img;
}

void get_rgb(const uint8_t *raw_rgb, int stride, double x, double y, uint8_t *rgb) {
    int r = static_cast<int>(y), c = static_cast<int>(x);
    rgb[0] = raw_rgb[r * stride + (3 * c)];
    rgb[1] = raw_rgb[r * stride + (3 * c) + 1];
    rgb[2] = raw_rgb[r * stride + (3 * c) + 2];
}

/*
 * return rgb point cloud. Store corresponding pixel coordinates (2d) and camera coordinates into
 * vector img_coords and cam_coords respectively.
 */
pcl::PointCloud<pcl::PointXYZRGB>::Ptr
get_point_cloud(const rs2::depth_frame &depth, const rs2::video_frame &rgb, vector<Point2d> &img_coords,
                vector<Point3d> &cam_coords) {
    rs2::pointcloud cloud;
    rs2::points points = cloud.calculate(depth);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    // Configure point cloud parameters
    auto sp = points.get_profile().as<rs2::video_stream_profile>();
    rgb_cloud->width = sp.width();
    rgb_cloud->height = sp.height();
    rgb_cloud->is_dense = false;
    rgb_cloud->points.reserve(points.size());
    img_coords.reserve(points.size());
    cam_coords.reserve(points.size());

    // Extract byte data from synchronized RGB and Infrared frames
    const auto *raw_rgb = (uint8_t *) rgb.get_data();
    int rgb_stride = rgb.get_stride_in_bytes();

    // Texture mapping
    cloud.map_to(rgb);
    points = cloud.calculate(depth);
    const rs2::vertex *vertices = points.get_vertices();
    const rs2::texture_coordinate *texture_coords = points.get_texture_coordinates();
    int w = rgb.get_width(), h = rgb.get_height();
    double x, y;
    uint8_t pixel[3];
    for (size_t i = 0; i != points.size(); ++i) {
        if (vertices[i].z != 0) {
            // Texture coordinates are theoretically in range [0, 1).
            x = texture_coords[i].u * static_cast<double>(w);
            y = texture_coords[i].v * static_cast<double>(h);
            /*
              Imagers for RGB and infrared are mounted at different places.
              Therefore, there exists places where infrared imager can see, but RGB imager cannot.
              We need to check if the projected u, v coordinates are out of bound.
            */
            if (x < 0 || x >= w || y < 0 || y >= h) {
                continue;
            }
            img_coords.push_back({x, y});
            cam_coords.push_back({vertices[i].x, -vertices[i].y, -vertices[i].z});
            get_rgb(raw_rgb, rgb_stride, x, y, pixel);
            // Add RGB point
            pcl::PointXYZRGB rgb_point;
            rgb_point.x = vertices[i].x;
            rgb_point.y = -vertices[i].y;
            rgb_point.z = -vertices[i].z;
            rgb_point.r = pixel[0];
            rgb_point.g = pixel[1];
            rgb_point.b = pixel[2];
            rgb_cloud->points.push_back(rgb_point);
        }
    }
    // Delete pre-allocated memory for discarded points
    rgb_cloud->points.shrink_to_fit();
    img_coords.shrink_to_fit();
    cam_coords.shrink_to_fit();
    return rgb_cloud;
}

/*
 * Extract and store information to View object for each selected time.
 */
vector<View> get_views(const string &fn, const vector<double> &time_elapsed) {
    vector<View> views;
    // Configure and initialize the pipe to read from recorded *.bag file
    rs2::config cfg;
    rs2::pipeline pipe;
    cfg.enable_device_from_file(fn);
    pipe.start(cfg);
    rs2::frameset data;
    double init_ts = -1; // A flag that is used to store initial timestamp later
    RotationEstimator re;
    auto curr = time_elapsed.begin();
    while (curr != time_elapsed.end()) {
        if (!pipe.poll_for_frames(&data)) {
            continue;
        }
        rs2::motion_frame gyro = data.first_or_default(RS2_STREAM_GYRO).as<rs2::motion_frame>();
        re.process(gyro.get_motion_data(), gyro.get_timestamp());
        if (init_ts == -1) {
            init_ts = data.get_timestamp();
            /* Every time when we reach the timestamp of interest,
             * Record the rotation matrix from last timestamp of interest to the current's.
             * Note: The rotation matrix stored in the first View is the rotation from initial frame to
             * the first timestamp of interest. It won't be used. The first rotation (first -> second)
             * will be stored in the second View.
             * Then reset the rotation estimator to start process the pairwise rotation to the next.
             */
        } else if (data.get_timestamp() >= init_ts + *curr) {
            rs2::video_frame rgb = data.get_color_frame();
            cv::Mat gray_img = frame_to_mat(rgb);
            rs2::depth_frame depth = data.get_depth_frame();
            vector<Point2d> img_coords;
            vector<Point3d> cam_coords;
            // Generate point cloud from RGB and depth frames. Store 2d image coordiante and 3d world coordinate for each point in the cloud.
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud = get_point_cloud(depth, rgb, img_coords, cam_coords);
            /* views[0].rotation is from initial frame to views[0]'s frame, which is not useful.
             * The first effective rotation matrix in stored in views[1].rotation which is from views[0] to views[1].
             */
            views.emplace_back(re.get_rotation_mat(), gray_img, img_coords, cam_coords, rgb_cloud);
            ++curr;
            re.reset();
            // Store the current timestamp and angular velocity as the new last_ts and angular_velocity
            re.process(gyro.get_motion_data(), gyro.get_timestamp());
        }
    }
    pipe.stop();
    return views;
}

/*
 * Map matched pixel location to image coordinates stored in img_coords.
 * Return the index of closest image coordinate. We can then use this index to find its corresponding
 * 3d camera coordinate (location of point in point cloud)
 */
size_t get_closest_idx(Point2d p, const vector<Point2d> &img_coords) {
    vector<double> norms;
    std::transform(img_coords.begin(), img_coords.end(), std::back_inserter(norms),
                   [p](Point2d p_) { return p.l2(p_); });
    return std::min_element(norms.cbegin(), norms.cend()) - norms.cbegin();
}

/*
 * Use SIFT feature detector and FLANN feature matcher to find matched pixel location between two
 * Views. We can use good matches to find the 3d point correspondences.
 */
Eigen::Matrix4d
estimate(const View &view1, const View &view2, int num_features, float ratio_thresh) {
    vector<pair<Point3d, Point3d>> correspondences;
    // Detect the key points using SIFT Detector, compute the descriptors.
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create(num_features);
    detector->detectAndCompute(view1.img, cv::noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(view2.img, cv::noArray(), keypoints2, descriptors2);
    // Matching descriptors with a FLANN based matcher
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<cv::DMatch> > knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
    // Filter matches using the Lowe's ratio test
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); ++i) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    // Visualize matches
    cv::Mat img_matches;
    drawMatches(view1.img, keypoints1, view2.img, keypoints2, good_matches, img_matches, cv::Scalar::all(-1),
                cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("Good Matches", img_matches);
    cv::waitKey();

    for (auto const m : good_matches) {
        cv::Point2f p1 = keypoints1[m.queryIdx].pt,
                p2 = keypoints2[m.trainIdx].pt;
        size_t idx_1 = get_closest_idx(Point2d{p1.x, p1.y}, view1.img_coords),
                idx_2 = get_closest_idx(Point2d{p2.x, p2.y}, view2.img_coords);
        correspondences.emplace_back(view1.cam_coords[idx_1], view2.cam_coords[idx_2]);
    }
    vector<int> removed_idx;
    Eigen::Matrix4d affine = robust_fitting(correspondences, view2.rotation, removed_idx);
    for (auto i :removed_idx) {
        good_matches.erase(good_matches.begin() + i);
    }
    drawMatches(view1.img, keypoints1, view2.img, keypoints2, good_matches, img_matches, cv::Scalar::all(-1),
                cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("Good Matches", img_matches);
    cv::waitKey();
    return affine;
}

/*
 * Use estimated rotation from IMU data. Estimate the translation, then return the affine matrix.
 * Note: Gyroscope is accurate only in short period of time. We need to carefully select frame to process.
 */
std::pair<Eigen::Matrix4d, vector<double>>
affine_estimation(vector<pair<Point3d, Point3d>> correspondences, Eigen::Matrix3d R) {
    size_t sz = correspondences.size();
    Eigen::MatrixXd C(3 * sz, 3);
    Eigen::VectorXd P(3 * sz);
    for (size_t i = 0; i != correspondences.size(); ++i) {
        Point3d p1 = correspondences[i].first, p2 = correspondences[i].second;
        C.row(3 * i) << 1, 0, 0;
        C.row(3 * i + 1) << 0, 1, 0;
        C.row(3 * i + 2) << 0, 0, 1;
        P(3 * i) = p2.x - R(0, 0) * p1.x - R(0, 1) * p1.y - R(0, 2) * p1.z;
        P(3 * i + 1) = p2.y - R(1, 0) * p1.x - R(1, 1) * p1.y - R(1, 2) * p1.z;
        P(3 * i + 2) = p2.z - R(2, 0) * p1.x - R(2, 1) * p1.y - R(2, 2) * p1.z;
    }
    Eigen::BDCSVD<Eigen::MatrixXd> svd(C, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd inv = svd.matrixV() * svd.singularValues().asDiagonal().inverse() * svd.matrixU().transpose();
    Eigen::Vector3d T = inv * P;
    Eigen::Matrix4d affine;
    Eigen::MatrixXd M(3, 4);
    affine << R(0, 0), R(0, 1), R(0, 2), T(0),
            R(1, 0), R(1, 1), R(1, 2), T(1),
            R(2, 0), R(2, 1), R(2, 2), T(2),
            0, 0, 0, 1;
    vector<double> errors;
    for (size_t i = 0; i != sz; ++i) {
        Point3d p1 = correspondences[i].first, p2 = correspondences[i].second;
        Eigen::Vector4d v1;
        v1 << p1.x, p1.y, p1.z, 1;
        Eigen::Vector4d v1_2 = affine * v1;
        Point3d p1_2 = {v1_2(0), v1_2(1), v1_2(2)};
        errors.push_back(p1_2.l2(p2));
        std::cout << p1 << "->" << p1_2 << " "
                  << p2 << " Error = " <<
                  p1_2.l2(p2) << std::endl;
    }
    return std::make_pair(affine, errors);
}

Eigen::Matrix4d
robust_fitting(vector<pair<Point3d, Point3d>> correspondences, Eigen::Matrix3d R, vector<int> &removed_idx) {
    Eigen::Matrix4d affine;
    double mean = -1;
    while (true) {
        auto res = affine_estimation(correspondences, R);
        vector<double> errors = res.second;
        size_t sz = errors.size();
        double curr_mean = std::accumulate(errors.begin(), errors.end(), 0.0) / sz;
        auto max_iter = std::max_element(errors.begin(), errors.end());
        if (mean == -1 || (mean > curr_mean && curr_mean > 0.01)) {
            correspondences.erase(correspondences.begin() + (max_iter - errors.begin()));
            removed_idx.push_back(max_iter - errors.begin());
            mean = curr_mean;
            affine = res.first;
            cout << "mean = " << curr_mean << ", max = " << *max_iter << "\n" << endl;
        } else {
            cout << "mean = " << curr_mean << ", max = " << *max_iter <<
                 "\n********************************************************************************************\n" << endl;

            return affine;
        }
    }
}

/*
 * FIXME: Currently, we apply the rotation R (prev->curr) to current points in order to
 *  map them onto the previous View coordinate system, which only works for two Views.
 *  If we have multiple view, it requires rotation matrix (first->curr) which is not pairwise.
 *  In addition, this approach will suffer from gyroscope drift.
 */
void combine_pc(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_ref, pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc,
                Eigen::Matrix4d affine) {
    pcl::transformPointCloud(*pc, *pc, affine);
    *pc_ref += *pc;
    pcl::visualization::PCLVisualizer::Ptr visualizer(new pcl::visualization::PCLVisualizer("3D Visualizer"));
    visualizer->setBackgroundColor(0, 0, 0);
    visualizer->addCoordinateSystem(0.5);
    visualizer->initCameraParameters();

    visualize_cloud(visualizer, pc_ref);
}

void visualize_cloud(pcl::visualization::PCLVisualizer::Ptr visualizer,
                     pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr rgb_cloud) {
    visualizer->removeAllShapes();
    visualizer->removeAllPointClouds();
    visualizer->addPointCloud<pcl::PointXYZRGB>(rgb_cloud, "RGB Cloud");
    visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "RGB Cloud");
    visualizer->resetCameraViewpoint("RGB Cloud");
    while (!visualizer->wasStopped()) {
        visualizer->spinOnce();
    }
}