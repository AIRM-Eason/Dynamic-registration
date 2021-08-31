#include "helper.h"

int main(int argc, char **argv) {

    // Select frames at following timestamps. Candidate frames should have decent exposure and focus.
    vector<double> time_elapsed{14500, 19000, 25000, 33000}; // in ms

    // Encapsulate all the required data in each candidate frames into View object.
    vector<View> views = get_views(argv[2], time_elapsed);

    // Pairwise scene reconstruction starting from the first candidate frame
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr reconstructed_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    *reconstructed_cloud += *views[0].rgb_cloud;
    for (size_t i = 1; i != views.size(); ++i) {
        Eigen::Matrix4d affine = estimate(views[i-1], views[i]);
        pcl::transformPointCloud(*reconstructed_cloud, *reconstructed_cloud, affine);
        *reconstructed_cloud += *views[i].rgb_cloud;
    }
    pcl::visualization::PCLVisualizer::Ptr visualizer(new pcl::visualization::PCLVisualizer("3D Visualizer"));
    visualizer->setBackgroundColor(0, 0, 0);
    visualizer->addCoordinateSystem(0.5);
    visualizer->initCameraParameters();
    visualize_cloud(visualizer, reconstructed_cloud);
}