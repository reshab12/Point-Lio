// #ifndef PARAM_H
// #define PARAM_H
#pragma once
#include <ros/ros.h>
#include <Eigen/Eigen>
#include <Eigen/Core>
#include <cstring>
#include "preprocess.h"
#include "IMU_Processing.h"
#include <sensor_msgs/NavSatFix.h>
#include <livox_ros_driver/CustomMsg.h>
#include <sensor_msgs/PointCloud2.h>
#include <mutex>
#include <omp.h>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <ivox/ivox3d.h>
#include <Python.h>
#include <condition_variable>
#include <sensor_msgs/Imu.h>
#include <pcl/common/transforms.h>
#include <geometry_msgs/Vector3.h>

// #define IVOX_NODE_TYPE_PHC

#ifdef IVOX_NODE_TYPE_PHC
    using IVoxType = faster_lio::IVox<3, faster_lio::IVoxNodeType::PHC, PointType>;
#else
    using IVoxType = faster_lio::IVox<3, faster_lio::IVoxNodeType::DEFAULT, PointType>;
#endif

extern bool is_first_frame;
extern double lidar_end_time, first_lidar_time, time_con;
extern double last_timestamp_lidar, last_timestamp_imu;
extern int pcd_index;
extern IVoxType::Options ivox_options_;
extern int ivox_nearby_type;
extern state_input state_in;
extern state_output state_out;
extern std::string lid_topic, imu_topic;
extern bool prop_at_freq_of_imu, check_satu, con_frame, cut_frame;
extern bool use_imu_as_input, space_down_sample;
extern bool extrinsic_est_en, publish_odometry_without_downsample;
extern int  init_map_size, con_frame_num;
extern double match_s, satu_acc, satu_gyro, cut_frame_time_interval;
extern float  plane_thr;
extern double filter_size_surf_min, filter_size_map_min, fov_deg;
// extern double cube_len; 
extern float  DET_RANGE;
extern bool   imu_en;
extern double imu_time_inte;
extern double laser_point_cov, acc_norm;
extern double acc_cov_input, gyr_cov_input, vel_cov;
extern double gyr_cov_output, acc_cov_output, b_gyr_cov, b_acc_cov;
extern double imu_meas_acc_cov, imu_meas_omg_cov; 
extern int    lidar_type, pcd_save_interval;
extern std::vector<double> gravity_init, gravity;
extern bool   runtime_pos_log, pcd_save_en, path_en;
extern bool   scan_pub_en, scan_body_pub_en;
extern shared_ptr<Preprocess> p_pre;
extern shared_ptr<ImuProcess> p_imu;
extern bool is_first_frame;

extern std::vector<double> extrinT;
extern std::vector<double> extrinR;
extern double time_diff_lidar_to_imu;
extern double lidar_time_inte, first_imu_time;
extern int cut_frame_num, orig_odom_freq;
extern double online_refine_time; //unit: s
extern bool cut_frame_init;
extern double time_update_last, time_current, time_predict_last_const, t_last;

/* DIFF-DRIVE MODEL: parameters of the differential-drive output filter
 * (namespace "diff_drive/" in the yaml file, see config/diff_drive.yaml). */
extern std::vector<double> dd_imu_extrinT, dd_imu_extrinR;     // IMU in base  (r_BI, R_BI)
extern std::vector<double> dd_lidar_extrinT, dd_lidar_extrinR; // LiDAR in base (t_BL, R_BL), optional
extern double dd_vel_cov, dd_acc_cov, dd_gyr_cov_xy, dd_gyr_cov_z;
extern bool   wheel_en, wheel_use_yaw_rate, wheel_gate_en;
extern std::string wheel_topic;
extern double wheel_vel_cov, wheel_omg_cov, wheel_vel_scale, wheel_max_delay;
extern bool   planar_prior_en, height_prior_en;
extern double planar_prior_std_deg, planar_prior_rate, height_prior_z0, height_prior_std;
// DIFF-DRIVE MODEL: debugging switches (namespace "diff_drive/debug/")
extern bool   dd_legacy_lidar_info_form, dd_jump_log;
extern double dd_jump_pos_thr, dd_jump_rot_thr_deg;

extern MeasureGroup Measures;

extern ofstream fout_out, fout_imu_pbp;
void readParameters(ros::NodeHandle &n);
void open_file();
Eigen::Matrix<double, 3, 1> SO3ToEuler(const SO3 &orient);
void reset_cov(Eigen::Matrix<double, 24, 24> & P_init);
// DIFF-DRIVE MODEL: initial covariance of the 17-DOF diff-drive state.
void reset_cov_output(Eigen::Matrix<double, DD_DIM_STATE, DD_DIM_STATE> & P_init_output);