#include "parameters.h"

bool is_first_frame = true;
double lidar_end_time = 0.0, first_lidar_time = 0.0, time_con = 0.0;
double last_timestamp_lidar = -1.0, last_timestamp_imu = -1.0;
int pcd_index = 0;
IVoxType::Options ivox_options_;
int ivox_nearby_type = 6;

std::vector<double> extrinT(3, 0.0);
std::vector<double> extrinR(9, 0.0);
state_input state_in;
state_output state_out;
std::string lid_topic, imu_topic;
bool prop_at_freq_of_imu = true, check_satu = true, con_frame = false, cut_frame = false;
bool use_imu_as_input = false, space_down_sample = true, publish_odometry_without_downsample = false;
int  init_map_size = 10, con_frame_num = 1;
double match_s = 81, satu_acc, satu_gyro, cut_frame_time_interval = 0.1;
float  plane_thr = 0.1f;
double filter_size_surf_min = 0.5, filter_size_map_min = 0.5, fov_deg = 180;
// double cube_len = 2000; 
float  DET_RANGE = 450;
bool   imu_en = true;
double imu_time_inte = 0.005;
double laser_point_cov = 0.01, acc_norm;
double vel_cov, acc_cov_input, gyr_cov_input;
double gyr_cov_output, acc_cov_output, b_gyr_cov, b_acc_cov;
double imu_meas_acc_cov, imu_meas_omg_cov; 
int    lidar_type, pcd_save_interval;
std::vector<double> gravity_init, gravity;
bool   runtime_pos_log, pcd_save_en, path_en, extrinsic_est_en = true;
bool   scan_pub_en, scan_body_pub_en;
shared_ptr<Preprocess> p_pre;
shared_ptr<ImuProcess> p_imu;
double time_update_last = 0.0, time_current = 0.0, time_predict_last_const = 0.0, t_last = 0.0;
double time_diff_lidar_to_imu = 0.0;

double lidar_time_inte = 0.1, first_imu_time = 0.0;
int cut_frame_num = 1, orig_odom_freq = 10;
double online_refine_time = 20.0; //unit: s
bool cut_frame_init = false; // true;

MeasureGroup Measures;

// DIFF-DRIVE MODEL: parameters (see parameters.h / config/diff_drive.yaml)
std::vector<double> dd_imu_extrinT(3, 0.0), dd_imu_extrinR{1, 0, 0, 0, 1, 0, 0, 0, 1};
std::vector<double> dd_lidar_extrinT, dd_lidar_extrinR;
double dd_vel_cov = 0.0, dd_acc_cov = 0.1, dd_gyr_cov_xy = 0.001, dd_gyr_cov_z = 0.1;
bool   wheel_en = false, wheel_use_yaw_rate = false, wheel_gate_en = true;
std::string wheel_topic;
double wheel_vel_cov = 0.0025, wheel_omg_cov = 0.01, wheel_vel_scale = 1.0, wheel_max_delay = 0.1;
bool   planar_prior_en = false, height_prior_en = false;
double planar_prior_std_deg = 1.5, planar_prior_rate = 20.0, height_prior_z0 = 0.0, height_prior_std = 0.05;
bool   dd_legacy_lidar_info_form = false, dd_jump_log = true;
double dd_jump_pos_thr = 0.05, dd_jump_rot_thr_deg = 1.0;

ofstream fout_out, fout_imu_pbp;

void readParameters(ros::NodeHandle &nh)
{
  p_pre.reset(new Preprocess());
  p_imu.reset(new ImuProcess());
  nh.param<bool>("prop_at_freq_of_imu", prop_at_freq_of_imu, 1);
  nh.param<bool>("use_imu_as_input", use_imu_as_input, 0);
  nh.param<bool>("check_satu", check_satu, 1);
  nh.param<int>("init_map_size", init_map_size, 100);
  nh.param<bool>("space_down_sample", space_down_sample, 1);
  nh.param<double>("mapping/satu_acc",satu_acc,3.0);
  nh.param<double>("mapping/satu_gyro",satu_gyro,35.0);
  nh.param<double>("mapping/acc_norm",acc_norm,1.0);
  nh.param<float>("mapping/plane_thr", plane_thr, 0.05f);
  nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
  nh.param<std::string>("common/lid_topic",lid_topic,"/livox/lidar");
  nh.param<std::string>("common/imu_topic", imu_topic,"/livox/imu");
  nh.param<bool>("common/con_frame",con_frame,false);
  nh.param<int>("common/con_frame_num",con_frame_num,1);
  nh.param<bool>("common/cut_frame",cut_frame,false);
  nh.param<double>("common/cut_frame_time_interval",cut_frame_time_interval,0.1);
  nh.param<double>("common/time_diff_lidar_to_imu",time_diff_lidar_to_imu,0.0);
  nh.param<double>("filter_size_surf",filter_size_surf_min,0.5);
  nh.param<double>("filter_size_map",filter_size_map_min,0.5);
  // nh.param<double>("cube_side_length",cube_len,2000);
  nh.param<float>("mapping/det_range",DET_RANGE,300.f);
  nh.param<double>("mapping/fov_degree",fov_deg,180);
  nh.param<bool>("mapping/imu_en",imu_en,true);
  nh.param<bool>("mapping/extrinsic_est_en",extrinsic_est_en,true);
  nh.param<double>("mapping/imu_time_inte",imu_time_inte,0.005);
  nh.param<double>("mapping/lidar_meas_cov",laser_point_cov,0.1);
  nh.param<double>("mapping/acc_cov_input",acc_cov_input,0.1);
  nh.param<double>("mapping/vel_cov",vel_cov,20);
  nh.param<double>("mapping/gyr_cov_input",gyr_cov_input,0.1);
  nh.param<double>("mapping/gyr_cov_output",gyr_cov_output,0.1);
  nh.param<double>("mapping/acc_cov_output",acc_cov_output,0.1);
  nh.param<double>("mapping/b_gyr_cov",b_gyr_cov,0.0001);
  nh.param<double>("mapping/b_acc_cov",b_acc_cov,0.0001);
  nh.param<double>("mapping/imu_meas_acc_cov",imu_meas_acc_cov,0.1);
  nh.param<double>("mapping/imu_meas_omg_cov",imu_meas_omg_cov,0.1);
  nh.param<double>("preprocess/blind", p_pre->blind, 1.0);
  nh.param<int>("preprocess/lidar_type", lidar_type, 1);
  nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
  nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
  nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, 1);
  nh.param<double>("mapping/match_s", match_s, 81);
  nh.param<std::vector<double>>("mapping/gravity", gravity, std::vector<double>());
  nh.param<std::vector<double>>("mapping/gravity_init", gravity_init, std::vector<double>());
  nh.param<std::vector<double>>("mapping/extrinsic_T", extrinT, std::vector<double>());
  nh.param<std::vector<double>>("mapping/extrinsic_R", extrinR, std::vector<double>());
  nh.param<bool>("odometry/publish_odometry_without_downsample", publish_odometry_without_downsample, false);
  nh.param<bool>("publish/path_en",path_en, true);
  nh.param<bool>("publish/scan_publish_en",scan_pub_en,1);
  nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en,1);
  nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
  nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
  nh.param<int>("pcd_save/interval", pcd_save_interval, -1);

  nh.param<double>("mapping/lidar_time_inte",lidar_time_inte,0.1);
  nh.param<double>("mapping/lidar_meas_cov",laser_point_cov,0.1);

  nh.param<float>("mapping/ivox_grid_resolution", ivox_options_.resolution_, 0.2);
  nh.param<int>("ivox_nearby_type", ivox_nearby_type, 18);
  if (ivox_nearby_type == 0) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
  } else if (ivox_nearby_type == 6) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
  } else if (ivox_nearby_type == 18) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
  } else if (ivox_nearby_type == 26) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
  } else {
    // LOG(WARNING) << "unknown ivox_nearby_type, use NEARBY18";
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
  }
    p_imu->gravity_ << VEC_FROM_ARRAY(gravity);

  /* DIFF-DRIVE MODEL: parameters of the differential-drive output filter.
   * Process noise defaults are derived from the original output-model values
   * so that existing configs keep comparable magnitudes (Q * dt^2 convention):
   *   acc_cov        : forward jerk           (was acc_cov_output, 3D)
   *   gyr_cov_z      : yaw angular accel.     (was gyr_cov_output, 3D)
   *   gyr_cov_xy     : roll/pitch angular acc. -> much smaller (terrain only) */
  nh.param<std::vector<double>>("diff_drive/extrinsic_T_imu_in_base", dd_imu_extrinT, std::vector<double>{0.0, 0.0, 0.0});
  nh.param<std::vector<double>>("diff_drive/extrinsic_R_imu_in_base", dd_imu_extrinR, std::vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1});
  nh.param<std::vector<double>>("diff_drive/extrinsic_T_lidar_in_base", dd_lidar_extrinT, std::vector<double>());
  nh.param<std::vector<double>>("diff_drive/extrinsic_R_lidar_in_base", dd_lidar_extrinR, std::vector<double>());
  nh.param<double>("diff_drive/vel_cov", dd_vel_cov, 0.0);
  nh.param<double>("diff_drive/acc_cov", dd_acc_cov, acc_cov_output);
  nh.param<double>("diff_drive/gyr_cov_z", dd_gyr_cov_z, gyr_cov_output);
  nh.param<double>("diff_drive/gyr_cov_xy", dd_gyr_cov_xy, 0.01 * gyr_cov_output);

  nh.param<bool>("diff_drive/wheel/enable", wheel_en, false);
  nh.param<std::string>("diff_drive/wheel/topic", wheel_topic, "/odom");
  nh.param<bool>("diff_drive/wheel/use_yaw_rate", wheel_use_yaw_rate, false);
  nh.param<bool>("diff_drive/wheel/chi2_gate_en", wheel_gate_en, true);
  nh.param<double>("diff_drive/wheel/vel_cov", wheel_vel_cov, 0.0025);
  nh.param<double>("diff_drive/wheel/omg_cov", wheel_omg_cov, 0.01);
  nh.param<double>("diff_drive/wheel/vel_scale", wheel_vel_scale, 1.0);
  nh.param<double>("diff_drive/wheel/max_delay", wheel_max_delay, 0.1);

  nh.param<bool>("diff_drive/planar_prior/enable", planar_prior_en, false);
  nh.param<double>("diff_drive/planar_prior/std_deg", planar_prior_std_deg, 1.5);
  nh.param<double>("diff_drive/planar_prior/rate", planar_prior_rate, 20.0);
  nh.param<bool>("diff_drive/height_prior/enable", height_prior_en, false);
  nh.param<double>("diff_drive/height_prior/z0", height_prior_z0, 0.0);
  nh.param<double>("diff_drive/height_prior/std", height_prior_std, 0.05);

  /* DIFF-DRIVE MODEL: debugging.
   *   legacy_lidar_info_form : restore the original weighting of the
   *       information-form LiDAR update (A/B test of the bug fix).
   *   jump_log               : warn whenever a single filter step moves the
   *       pose by more than jump_pos_thr [m] or jump_rot_thr_deg [deg]. */
  nh.param<bool>("diff_drive/debug/legacy_lidar_info_form", dd_legacy_lidar_info_form, false);
  nh.param<bool>("diff_drive/debug/jump_log", dd_jump_log, true);
  nh.param<double>("diff_drive/debug/jump_pos_thr", dd_jump_pos_thr, 0.05);
  nh.param<double>("diff_drive/debug/jump_rot_thr_deg", dd_jump_rot_thr_deg, 1.0);

  if (dd_imu_extrinT.size() != 3 || dd_imu_extrinR.size() != 9)
  {
    ROS_ERROR("diff_drive/extrinsic_[T|R]_imu_in_base must have 3 / 9 entries, using identity.");
    dd_imu_extrinT = std::vector<double>{0.0, 0.0, 0.0};
    dd_imu_extrinR = std::vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1};
  }
  if (!dd_lidar_extrinT.empty() && (dd_lidar_extrinT.size() != 3 || dd_lidar_extrinR.size() != 9))
  {
    ROS_ERROR("diff_drive/extrinsic_[T|R]_lidar_in_base must have 3 / 9 entries, deriving it from the LiDAR-IMU extrinsic instead.");
    dd_lidar_extrinT.clear();
    dd_lidar_extrinR.clear();
  }
}

Eigen::Matrix<double, 3, 1> SO3ToEuler(const SO3 &rot) 
{
    double sy = sqrt(rot(0,0)*rot(0,0) + rot(1,0)*rot(1,0));
    bool singular = sy < 1e-6;
    double x, y, z;
    if(!singular)
    {
        x = atan2(rot(2, 1), rot(2, 2));
        y = atan2(-rot(2, 0), sy);   
        z = atan2(rot(1, 0), rot(0, 0));  
    }
    else
    {    
        x = atan2(-rot(1, 2), rot(1, 1));    
        y = atan2(-rot(2, 0), sy);    
        z = 0;
    }
    Eigen::Matrix<double, 3, 1> ang(x, y, z);
    return ang;
}

void open_file()
{

    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
    fout_imu_pbp.open(DEBUG_FILE_DIR("imu_pbp.txt"),ios::out);
    if (fout_out && fout_imu_pbp)
        cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
    else
        cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

}

void reset_cov(Eigen::Matrix<double, 24, 24> & P_init)
{
    P_init = MD(24, 24)::Identity() * 0.1;
    P_init.block<3, 3>(21, 21) = MD(3,3)::Identity() * 0.0001;
    P_init.block<6, 6>(15, 15) = MD(6,6)::Identity() * 0.001;
}

// DIFF-DRIVE MODEL: initial covariance of the 17-DOF diff-drive state.
// Same magnitudes as the original output model: 0.01 for pose / motion states,
// 0.001 for the IMU biases (the former gravity and extrinsic blocks are gone).
void reset_cov_output(Eigen::Matrix<double, DD_DIM_STATE, DD_DIM_STATE> & P_init_output)
{
    P_init_output = MD(DD_DIM_STATE, DD_DIM_STATE)::Identity() * 0.01;
    P_init_output.block<6, 6>(DD_IDX_BG, DD_IDX_BG) = MD(6,6)::Identity() * 0.001;
}