#include <opencv2/opencv.hpp>

#ifndef MONOCULAR_CALIBRATION_CAMERACALIBRATION_H
#define MONOCULAR_CALIBRATION_CAMERACALIBRATION_H

using namespace std;
using namespace cv;

class CameraCalibration {
private:
    vector<cv::Mat> H_collection;
    cv::Mat normalize_matrix;
public:
    vector<string> file_names;
    int image_num;
    int board_width, board_height;
    cv::Size board_size;
    int board_num;
    cv::Size image_size;

    vector<vector<cv::Point2d> > image_points;
    vector<vector<cv::Point2d>> image_points_proj;
    vector<vector<cv::Point2d>> image_points_norm;
    vector<vector<cv::Point3d> > object_points;
    cv::Mat intrinsic_matrix, dist_coeffs;
    vector<cv::Mat> tvecs, rvecs;
    vector<cv::Mat> Rts;

    CameraCalibration(int board_w, int board_h, string &file_path);

    void LoadImages();

    void NormalizeImagesPoints();

    double Calibrate();

    double CalReprojectionError();

    void Dedestrotion();

};


#endif //MONOCULAR_CALIBRATION_CAMERACALIBRATION_H
