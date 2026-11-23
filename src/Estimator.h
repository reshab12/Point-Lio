#ifndef Estimator_H
#define Estimator_H

#include "common_lib.h"
#include "parameters.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
// #include <ikd-Tree/ikd_Tree.h>
#include <pcl/io/pcd_io.h>
#include <unordered_set>

extern PointCloudXYZI::Ptr normvec; //(new PointCloudXYZI(100000, 1));
extern std::vector<int> time_seq;
extern PointCloudXYZI::Ptr feats_down_body; //(new PointCloudXYZI());
extern PointCloudXYZI::Ptr feats_down_world; //(new PointCloudXYZI());
extern std::vector<V3D> pbody_list;
extern std::vector<PointVector> Nearest_Points; 
extern std::shared_ptr<IVoxType> ivox_;                    // localmap in ivox
extern std::vector<float> pointSearchSqDis;
extern bool point_selected_surf[100000]; // = {0};
extern std::vector<M3D> crossmat_list;
extern int effct_feat_num;
extern int k;
extern int idx;
extern V3D angvel_avr, acc_avr, acc_avr_norm;
extern int feats_down_size;
// extern std::vector<Eigen::Vector3d> normvec_holder;
extern V3D Lidar_T_wrt_IMU; //(Zero3d);
extern M3D Lidar_R_wrt_IMU; //(Eye3d);
extern double G_m_s2;
extern input_ikfom input_in;

/* DIFF-DRIVE MODEL: frames and extrinsics
 *   "Body" = the frame the filter state refers to.
 *     - IMU-as-input filter (kf_input, original Point-LIO): Body = IMU frame I.
 *     - IMU-as-output filter (kf_output, diff-drive model): Body = robot base B
 *       (base_link, origin at the midpoint of the drive axle, x forward, z up).
 *   Lidar_R_wrt_Body / Lidar_T_wrt_Body : LiDAR -> Body extrinsic used for every
 *       LiDAR point (R_BL, t_BL in the note). For kf_input it equals
 *       Lidar_R_wrt_IMU / Lidar_T_wrt_IMU, so the original behaviour is unchanged.
 *   Imu_R_wrt_Base / Imu_T_wrt_Base : IMU -> base extrinsic (R_BI, r_BI in the
 *       note), r_BI = lever arm of the IMU expressed in B.
 *   gravity_world : fixed gravity vector in the gravity-aligned world frame G
 *       (replaces the former 'gravity' state of the output filter). */
extern V3D Lidar_T_wrt_Body;
extern M3D Lidar_R_wrt_Body;
extern V3D Imu_T_wrt_Base;
extern M3D Imu_R_wrt_Base;
extern V3D gravity_world;

Eigen::Matrix<double, 24, 24> process_noise_cov_input();

// DIFF-DRIVE MODEL: 17 x 17 process noise of the diff-drive state.
Eigen::Matrix<double, DD_DIM_STATE, DD_DIM_STATE> process_noise_cov_output();

//double L_offset_to_I[3] = {0.04165, 0.02326, -0.0284}; // Avia 
//vect3 Lidar_offset_to_IMU(L_offset_to_I, 3);
Eigen::Matrix<double, 24, 1> get_f_input(state_input &s, const input_ikfom &in);

// DIFF-DRIVE MODEL: process function f(x, 0) of eq. (5) (17 x 1).
Eigen::Matrix<double, DD_DIM_STATE, 1> get_f_output(state_output &s, const input_ikfom &in);

Eigen::Matrix<double, 24, 24> df_dx_input(state_input &s, const input_ikfom &in);

// Eigen::Matrix<double, 24, 12> df_dw_input(state_input &s, const input_ikfom &in);

// DIFF-DRIVE MODEL: R^n part of df/dx (17 x 17), see eq. (20).
Eigen::Matrix<double, DD_DIM_STATE, DD_DIM_STATE> df_dx_output(state_output &s, const input_ikfom &in);

// Eigen::Matrix<double, 30, 15> df_dw_output(state_output &s);

void h_model_input(state_input &s, Eigen::Matrix3d cov_p, Eigen::Matrix3d cov_R, esekfom::dyn_share_modified<double> &ekfom_data);

void h_model_output(state_output &s, Eigen::Matrix3d cov_p, Eigen::Matrix3d cov_R, esekfom::dyn_share_modified<double> &ekfom_data);

void h_model_IMU_output(state_output &s, esekfom::dyn_share_modified<double> &ekfom_data);

/* DIFF-DRIVE MODEL: additional (optional) measurements of the output filter.
 *   update_wheel_output       : wheel odometry (v, omega_z), note sec. 4.3.
 *                               Returns false if the chi^2 gate rejected it.
 *   update_planar_prior_output: soft "body z || world z" prior, note sec. 4.4,
 *                               plus optional height prior p_z = z0. */
bool update_wheel_output(double v_meas, double omega_z_meas);
void update_planar_prior_output();

// DIFF-DRIVE MODEL: helpers to derive quantities that are no longer states.
V3D dd_world_velocity(const state_output &s);   // G_v = R e1 v
V3D dd_body_acceleration(const state_output &s); // B_a_B = [a, w_z v, -w_y v]  (eq. (3))

void pointBodyToWorld(PointType const * const pi, PointType * const po);

#endif