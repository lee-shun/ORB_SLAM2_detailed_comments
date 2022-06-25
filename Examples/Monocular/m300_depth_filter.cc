/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: m300_depth_filter.cc
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2022-06-02
 *
 *   @Description:
 *
 *******************************************************************************/

#include "./m300_depth_filter_tools/system_lib.h"
#include "./m300_depth_filter_tools/FileWritter.hpp"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <iomanip>

#include <opencv2/core/core.hpp>
#include <unistd.h>
#include "System.h"

using namespace std;

/**
 * @brief 获取图像序列中每一张图像的访问路径和时间戳
 * @param[in]  strSequence              图像序列的存放路径
 * @param[out] vstrImageFilenames       图像序列中每张图像的存放路径
 * @param[out] vTimestamps              图像序列中每张图像的时间戳
 */
void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

bool GetAllImageNames(const string &strSequence,
                      std::vector<std::string> &all_image_names,
                      vector<double> &vTimestamps) {
  size_t image_num = pose_correction::modules::GetFileNum(strSequence);

  all_image_names.clear();
  all_image_names.reserve(image_num);

  for (size_t i = 0; i < image_num; ++i) {
    std::string img_name = strSequence + "/" + std::to_string(i) + ".png";
    all_image_names.push_back(img_name);
    vTimestamps.push_back(static_cast<double>(i));
  }
  return true;
}

/**
 * TODO: need each frame and its R and t
 * */
int main(int argc, char **argv) {
  if (argc != 4) {
    cerr << endl
         << "Usage: ./M300_depth_filter path_to_vocabulary path_to_settings "
            "path_to_sequence"
         << endl;
    return 1;
  }

  // Retrieve paths to images
  vector<string> vstrImageFilenames;
  vector<double> vTimestamps;
  GetAllImageNames(string(argv[3]), vstrImageFilenames, vTimestamps);

  int nImages = vstrImageFilenames.size();

  // Create SLAM system. It initializes all system threads and gets ready to
  // process frames.
  ORB_SLAM2::System SLAM(argv[1], argv[2], ORB_SLAM2::System::MONOCULAR, true);

  // Vector for tracking time statistics
  vector<float> vTimesTrack;
  vTimesTrack.resize(nImages);

  cout << endl << "-------" << endl;
  cout << "Start processing sequence ..." << endl;
  cout << "Images in the sequence: " << nImages << endl << endl;

  // Main loop
  cv::Mat im;
  FFDS::TOOLS::FileWritter pose_writer("Monoslam_pose.csv", 9);
  pose_writer.new_open();
  for (int ni = 0; ni < nImages; ni++) {
    // Read image from file
    im = cv::imread(vstrImageFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);
    double tframe = vTimestamps[ni];

    if (im.empty()) {
      cerr << endl
           << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
      return 1;
    }

#ifdef COMPILEDWITHC11
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point t1 =
        std::chrono::monotonic_clock::now();
#endif

    // Pass the image to the SLAM system
    cv::Mat p = SLAM.TrackMonocular(im, tframe);

    std::cout << "pose" << p << std::endl;
    if (ORB_SLAM2::Tracking::eTrackingState::OK == SLAM.GetTrackingState()) {
      pose_writer.write(ni, p.at<float>(0, 0), p.at<float>(0, 1),
                        p.at<float>(0, 2), p.at<float>(0, 3), p.at<float>(1, 0),
                        p.at<float>(1, 1), p.at<float>(1, 2), p.at<float>(1, 3),
                        p.at<float>(2, 0), p.at<float>(2, 1), p.at<float>(2, 2),
                        p.at<float>(2, 3));
    }

#ifdef COMPILEDWITHC11
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point t2 =
        std::chrono::monotonic_clock::now();
#endif

    double ttrack =
        std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1)
            .count();

    vTimesTrack[ni] = ttrack;

    // Wait to load the next frame
    double T = 0;
    if (ni < nImages - 1)
      T = vTimestamps[ni + 1] - tframe;
    else if (ni > 0)
      T = tframe - vTimestamps[ni - 1];

    if (ttrack < T) usleep((T - ttrack) * 1e6);
  }

  // Stop all threads
  SLAM.Shutdown();

  // Tracking time statistics
  sort(vTimesTrack.begin(), vTimesTrack.end());
  float totaltime = 0;
  for (int ni = 0; ni < nImages; ni++) {
    totaltime += vTimesTrack[ni];
  }
  cout << "-------" << endl << endl;
  cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
  cout << "mean tracking time: " << totaltime / nImages << endl;

  // Save camera trajectory
  SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectoryM300DepthFilter.txt");

  return 0;
}

// 获取图像序列中每一张图像的访问路径和时间戳
void LoadImages(const string &strPathToSequence,
                vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps) {
  // step 1 读取时间戳文件
  ifstream fTimes;
  string strPathTimeFile = strPathToSequence + "/times.txt";
  fTimes.open(strPathTimeFile.c_str());
  while (!fTimes.eof()) {
    string s;
    getline(fTimes, s);
    // 当该行不为空的时候执行
    if (!s.empty()) {
      stringstream ss;
      ss << s;
      double t;
      ss >> t;
      // 保存时间戳
      vTimestamps.push_back(t);
    }
  }

  // step 1 使用左目图像, 生成左目图像序列中的每一张图像的文件名
  string strPrefixLeft = strPathToSequence + "/image_0/";

  const int nTimes = vTimestamps.size();
  vstrImageFilenames.resize(nTimes);

  for (int i = 0; i < nTimes; i++) {
    stringstream ss;
    ss << setfill('0') << setw(6) << i;
    vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
  }
}
