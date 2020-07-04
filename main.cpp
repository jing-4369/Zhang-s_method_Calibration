#include <iostream>
#include <opencv2/opencv.hpp>
#include "CameraCalibration/CameraCalibration.h"

using namespace std;

int main(int argc, char **argv) {

    const string keys =
            "{file_path||directory for store images}"
            "{board_width|6|width of board}"
            "{board_height|9|height of board}";
    cv::CommandLineParser parser(argc, argv, keys);
    int board_width = parser.get<int>("board_width");
    int board_height = parser.get<int>("board_height");
    string file_path = parser.get<string>("file_path");
    CameraCalibration my_camera(board_width, board_height, file_path);
    double error = my_camera.Calibrate();
    cout << "----- DONE! ----- Reprojection error is " << error << endl;
    cout << "\nimage width:" << static_cast<int>(my_camera.image_size.width);
    cout << "\nimage height: " << static_cast<int>(my_camera.image_size.height);
    cout << "\nintrinsic matrix:" << my_camera.intrinsic_matrix;
    cout << "\ndistortion coefficients: " << my_camera.dist_coeffs << "\n" << endl;

    my_camera.Dedestrotion();
    return 0;
}