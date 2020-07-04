#include "CameraCalibration.h"

#define TOL 1.E-8

static void cholesky_decomposition(const Mat &A, Mat &L);

CameraCalibration::CameraCalibration(int board_w, int board_h, string &file_path) {
    board_width = board_w;
    board_height = board_h;
    board_num = board_height * board_width;
    board_size = cv::Size(board_width, board_height);

    cv::glob(file_path, file_names);
    image_num = file_names.size();
    LoadImages();
    NormalizeImagesPoints();
}

void CameraCalibration::LoadImages() {
    for (size_t i = 0; i < file_names.size(); i++) {
        cv::Mat image = cv::imread(file_names[i]);
        if (image.empty()) {
            cerr << file_names[i] << "is empty!" << endl;
        }
        image_size = image.size();
        vector<cv::Point2d> corners;
        bool found = cv::findChessboardCorners(image, board_size, corners);

        if (found) {

            image_points.push_back(corners);
            object_points.emplace_back();
            vector<cv::Point3d> &opts = object_points.back();

            opts.resize(board_num);
            for (int j = 0; j < board_num; j++) {
                opts[j] = cv::Point3d(static_cast<double>(j / board_width),
                                      static_cast<double>(j % board_width), 0.0f);
            }
        }
        cout << "Found" << image_points.size() << "total boards."
             << "This one from chessboard iamge #" << i + 1 << "," << file_names[i] << endl;
    }
}

void CameraCalibration::NormalizeImagesPoints() {
    double sx = 2.0 / image_size.width;
    double sy = 2.0 / image_size.height;
    double x0 = image_size.width / 2;
    double y0 = image_size.height / 2;
    for (int i = 0; i < image_num; i++) {
        vector<cv::Point2d> points_norm;
        for (int j = 0; j < board_num; j++) {
            points_norm.push_back(
                    cv::Point2d(sx * (image_points[i][j].x - x0), sy * (image_points[i][j].y - y0)));
        }
        image_points_norm.push_back(points_norm);
    }
    normalize_matrix = (cv::Mat_<double>(3, 3) << sx, 0, -sx * x0, 0, sy, -sy * y0, 0, 0, 1);
}

double CameraCalibration::Calibrate() {
    cout << "----- CALIBRATING -----" << endl;
    /* 1. close-form solution
    * step 1: solve H for every image(3*3), using SVD decomposition
            *
    * step 2: solve B =  A^(-T)*A
            *
    * step 3: solve every image R & T
            *
    * step 4: solve dist_coeffs, using least squre
    */

    //step 1:solve H
    for (int i = 0; i < image_num; i++) {
        cv::Mat A(board_num * 2, 9, CV_64FC1);
        for (int j = 0; j < board_num; j++) {
            double x = image_points_norm[i][j].x;
            double y = image_points_norm[i][j].y;
            cv::Point3d X = object_points[i][j];
            X.z = 1;
            vector<cv::Mat> mats = {cv::Mat(X).t(),
                                    cv::Mat::zeros(1, 3, CV_64FC1),
                                    cv::Mat(-x * X).t()};
            cv::hconcat(mats, A.row(2 * j));
            normalize(A.row(2 * j), A.row(2 * j), 1);

            mats = {cv::Mat::zeros(1, 3, CV_64FC1),
                    cv::Mat(X).t(),
                    cv::Mat(-y * X).t()};
            cv::hconcat(mats, A.row(2 * j + 1));
            normalize(A.row(2 * j + 1), A.row(2 * j + 1), 1);
        }
        cv::Mat w, V, D;
        cv::SVDecomp(A, w, V, D);
        Mat H = D.t().col(D.size().width - 1);
        Mat temp = H.clone();
        H = temp.reshape(0, 3);
        H = H.t();
        H = (1 / H.at<double>(2, 2)) * H;
        H_collection.push_back(H);
    }
    //step 2:solve B & A
    cv::Mat C(2 * image_num, 6, CV_64FC1);
    for (int i = 0; i < image_num; i++) {
        cv::Mat H = H_collection[i];
        cv::Mat v11 = (cv::Mat_<double>(1, 6) <<
                                              H.at<double>(0, 0) * H.at<double>(0, 0),
                H.at<double>(0, 0) * H.at<double>(0, 1) + H.at<double>(0, 1) * H.at<double>(0, 0),
                H.at<double>(0, 1) * H.at<double>(0, 1),
                H.at<double>(0, 2) * H.at<double>(0, 0) + H.at<double>(0, 0) * H.at<double>(0, 2),
                H.at<double>(0, 2) * H.at<double>(0, 1) + H.at<double>(0, 1) * H.at<double>(0, 2),
                H.at<double>(0, 2) * H.at<double>(0, 2));
        cv::Mat v22 = (cv::Mat_<double>(1, 6) <<
                                              H.at<double>(1, 0) * H.at<double>(1, 0),
                H.at<double>(1, 0) * H.at<double>(1, 1) + H.at<double>(1, 1) * H.at<double>(1, 0),
                H.at<double>(1, 1) * H.at<double>(1, 1),
                H.at<double>(1, 2) * H.at<double>(1, 0) + H.at<double>(1, 0) * H.at<double>(1, 2),
                H.at<double>(1, 2) * H.at<double>(1, 1) + H.at<double>(1, 1) * H.at<double>(1, 2),
                H.at<double>(1, 2) * H.at<double>(1, 2));
        cv::Mat v12 = (cv::Mat_<double>(1, 6) <<
                                              H.at<double>(0, 0) * H.at<double>(1, 0),
                H.at<double>(0, 0) * H.at<double>(1, 1) + H.at<double>(1, 0) * H.at<double>(0, 1),
                H.at<double>(0, 1) * H.at<double>(1, 1),
                H.at<double>(0, 2) * H.at<double>(1, 0) + H.at<double>(0, 0) * H.at<double>(1, 2),
                H.at<double>(0, 2) * H.at<double>(1, 1) + H.at<double>(0, 1) * H.at<double>(1, 2),
                H.at<double>(0, 2) * H.at<double>(1, 2));
        cv::Mat v_12 = v11 - v22;
        normalize(v_12, v_12, 1);
        normalize(v12, v12, 1);
        v12.copyTo(C.row(2 * i));
        v_12.copyTo(C.row(2 * i + 1));
    }
    cv::Mat w, V, D;
    cv::SVDecomp(C, w, V, D);
    cv::Mat b = D.t().col(D.size().width - 1);
    cv::Mat B = (cv::Mat_<double>(3, 3) <<
                                        b.at<double>(0, 0), b.at<double>(1, 0), b.at<double>(3, 0),
            b.at<double>(1, 0), b.at<double>(2, 0), b.at<double>(4, 0),
            b.at<double>(3, 0), b.at<double>(4, 0), b.at<double>(5, 0));

    // solve A using B
    double *B1 = B.ptr<double>(0);
    double *B2 = B.ptr<double>(1);
    double *B3 = B.ptr<double>(2);
    double alpha, betha, gamma, u0, v0, lambda;

    double den = B1[0] * B2[1] - B1[1] * B1[1];
    v0 = (B1[1] * B1[2] - B1[0] * B2[2]) / den;
    lambda = B3[2] - (B1[2] * B1[2] + v0 * (B1[1] * B1[2] - B1[0] * B2[2])) / B1[0];
    alpha = sqrt(lambda / B1[0]);
    betha = sqrt(lambda * B1[0] / den);
    gamma = -B1[1] * alpha * alpha * betha / lambda;
    u0 = gamma * v0 / betha - B1[2] * alpha * alpha / lambda;

    cv::Mat A = (cv::Mat_<double>(3, 3) <<
                                        alpha, gamma, u0,
            0, betha, v0,
            0, 0, 1);
    intrinsic_matrix = normalize_matrix.inv() * A;
    cout << "intrinsic_matrix:" << endl;
    cout << intrinsic_matrix << endl;

    //step 3: solve R & T
    for (int i = 0; i < image_num; i++) {
        double lambda1, lambda2, lambda3;
        cv::Mat H = H_collection[i];
        H = normalize_matrix.inv() * H.t();
        cv::Mat r1 = intrinsic_matrix.inv() * H.col(0);
        cv::Mat r2 = intrinsic_matrix.inv() * H.col(1);
        cv::Mat t = intrinsic_matrix.inv() * H.col(2);
        cv::Mat r3 = r1.cross(r2);
        lambda1 = norm(r1, NORM_L2);
        lambda2 = norm(r2, cv::NORM_L2);
        lambda3 = (lambda1 + lambda2) / 2;
        normalize(r1, r1, 1);
        normalize(r2, r2, 1);
        normalize(r3, r3, 1);
        t = 1. / lambda3 * t;
        Mat R(3, 3, CV_64FC1);
        r1.copyTo(R.col(0));
        r2.copyTo(R.col(1));
        r3.copyTo(R.col(2));
        rvecs.push_back(R);
        tvecs.push_back(t);
        cv::Mat Rt(3, 4, CV_64FC1);
        hconcat(R, t, Rt);
        Rts.push_back(Rt);
    }

    //step 4: solve distortion coefficients
    for (int i = 0; i < image_num; i++) {
        cv::Mat Rt = Rts[i];
        vector<Point2d> proj_points;
        for (int j = 0; j < board_num; j++) {
            Point3d X = object_points[i][j];
            Mat x = (Mat_<double>(4, 1) << X.x, X.y, X.z, 1);
            cv::Mat p = intrinsic_matrix * Rt * x;
            double u = p.at<double>(0, 0) / p.at<double>(2, 0);
            double v = p.at<double>(1, 0) / p.at<double>(2, 0);
            proj_points.push_back(Point2d(u, v));
        }
        image_points_proj.push_back(proj_points);
    }

    cv::Mat D_(2 * image_num * board_num, 2, CV_64FC1);
    cv::Mat d_(2 * image_num * board_num, 1, CV_64FC1);
    double *ptr_D = D_.ptr<double>(0);
    double *ptr_d = d_.ptr<double>(0);
    for (int i = 0; i < image_num; i++) {
        for (int j = 0; j < board_num; j++) {
            double u = image_points[i][j].x;
            double v = image_points[i][j].y;
            double u0 = image_size.width / 2;
            double v0 = image_size.height / 2;
            double r_2 = image_points_norm[i][j].x * image_points_norm[i][j].x
                         + image_points_norm[i][j].y * image_points_norm[i][j].y;
            double u_prime = image_points_proj[i][j].x;
            double v_prime = image_points_proj[i][j].y;

            ptr_D[0] = (u - u0) * r_2;
            ptr_D[1] = (u - u0) * r_2 * r_2;
            ptr_D[2] = (v - v0) * r_2;
            ptr_D[3] = (v - v0) * r_2 * r_2;
            ptr_D = ptr_D + 4;
            ptr_d[0] = u_prime - u;
            ptr_d[1] = v_prime - v;
            ptr_d = ptr_d + 2;
        }

    }
    cv::Mat K = (D_.t() * D_).inv() * D_.t() * d_;
    hconcat(K.t(), Mat::zeros(1, 2, CV_64FC1), dist_coeffs);

    double error = CalReprojectionError();
    return error;
    /*  2. optimizing solution
            * step 5 : optimizing all parameters using maximum likelihood estimation
    */
}

double CameraCalibration::CalReprojectionError() {
    double mean_error = 0;
    for (int i = 0; i < image_num; i++) {
        Mat Rt = Rts[i];
        for (int j = 0; j < board_num; j++) {
            Point3d point = object_points[i][j];
            Mat X = (Mat_<double>(4, 1) << point.x, point.y, point.z, 1);
            Mat camera_point = Rt * X;
            double x = camera_point.at<double>(0, 0) / camera_point.at<double>(2, 0);
            double y = camera_point.at<double>(1, 0) / camera_point.at<double>(2, 0);
            double r_2 = x * x + y * y;
            double dist = 1 + dist_coeffs.at<double>(0, 0) * r_2 + dist_coeffs.at<double>(0, 1) * r_2 * r_2;
            double x_correct = x / dist;
            double y_correct = y / dist;
            Mat camera_coordinate = (Mat_<double>(3, 1) << x_correct, y_correct, 1);
            Mat u_p = intrinsic_matrix * camera_coordinate;
            double ux_p = u_p.at<double>(0, 0) / u_p.at<double>(2, 0);
            double uy_p = u_p.at<double>(1, 0) / u_p.at<double>(2, 0);

            Point2d u = image_points[i][j];
            double error = sqrt((u.x - ux_p) * (u.x - ux_p) + (u.y - uy_p) * (u.y - uy_p));
            mean_error = mean_error + error;
        }
    }
    mean_error = mean_error / (image_num * board_num);
    return mean_error;
}

void CameraCalibration::Dedestrotion() {
    cv::Mat map1, map2;
    cv::initUndistortRectifyMap(intrinsic_matrix, dist_coeffs, cv::Mat(),
                                intrinsic_matrix, image_size, CV_16SC2, map1, map2);
    for (const auto &file_name : file_names) {
        cv::Mat image;
        cv::Mat image0 = cv::imread(file_name);
        if (image0.empty()) {
            cerr << file_name << " is empty!" << endl;
            continue;
        }
        cv::remap(image0, image, map1, map2, cv::INTER_LINEAR,
                  cv::BORDER_CONSTANT, cv::Scalar());
        cv::undistort(image0, image, intrinsic_matrix, dist_coeffs);
        cv::imshow("Original", image0);
        cv::imshow("Undistorted", image);
        cv::waitKey();
    }

}


static void cholesky_decomposition(const Mat &A, Mat &L) {
    L = Mat::zeros(A.size(), CV_32F);
    int rows = A.rows;

    for (int i = 0; i < rows; ++i) {
        int j;
        double sum;

        for (j = 0; j < i; ++j)// (i>j)
        {
            sum = 0;
            for (int k = 0; k < j; ++k) {
                sum += L.at<double>(i, k) * L.at<double>(j, k);
            }
            L.at<double>(i, j) = (A.at<double>(i, j) - sum) / L.at<double>(j, j);
        }
        sum = 0;
        assert(i == j);
        for (int k = 0; k < j; ++k)//i == j
        {
            sum += L.at<double>(j, k) * L.at<double>(j, k);
        }
        L.at<double>(j, j) = sqrt(A.at<double>(j, j) - sum);

    }

}

