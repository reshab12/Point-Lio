// #include <../include/IKFoM/IKFoM_toolkit/esekfom/esekfom.hpp>
#include "Estimator.h"

PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
std::vector<int> time_seq;
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI(10000, 1));
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI(10000, 1));
std::vector<V3D> pbody_list;
std::vector<PointVector> Nearest_Points; 
std::shared_ptr<IVoxType> ivox_ = nullptr;                    // localmap in ivox
std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
bool   point_selected_surf[100000] = {0};
std::vector<M3D> crossmat_list;
int effct_feat_num = 0;
int k = 0;
int idx = -1;
esekfom::esekf<state_input, 24, input_ikfom> kf_input;
esekfom::esekf<state_output, DD_DIM_STATE, input_ikfom> kf_output; // DIFF-DRIVE MODEL: 17 DOF
input_ikfom input_in;
V3D angvel_avr, acc_avr, acc_avr_norm;
int feats_down_size = 0;
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);
double G_m_s2 = 9.81;

// DIFF-DRIVE MODEL: extrinsics and fixed gravity (see Estimator.h).
V3D Lidar_T_wrt_Body(Zero3d);
M3D Lidar_R_wrt_Body(Eye3d);
V3D Imu_T_wrt_Base(Zero3d);
M3D Imu_R_wrt_Base(Eye3d);
V3D gravity_world(0.0, 0.0, -9.81);

namespace {
// DIFF-DRIVE MODEL: skew-symmetric matrix  skew(a) * b = a x b.
inline M3D skew(const V3D &a)
{
	M3D m;
	m <<   0.0, -a(2),  a(1),
	      a(2),   0.0, -a(0),
	     -a(1),  a(0),   0.0;
	return m;
}
const V3D e1_(1.0, 0.0, 0.0);
const V3D e3_(0.0, 0.0, 1.0);
}

Eigen::Matrix<double, 24, 24> process_noise_cov_input()
{
	Eigen::Matrix<double, 24, 24> cov;
	cov.setZero();
	cov.block<3, 3>(3, 3).diagonal() << gyr_cov_input, gyr_cov_input, gyr_cov_input;
	cov.block<3, 3>(12, 12).diagonal() << acc_cov_input, acc_cov_input, acc_cov_input;
	cov.block<3, 3>(15, 15).diagonal() << b_gyr_cov, b_gyr_cov, b_gyr_cov;
	cov.block<3, 3>(18, 18).diagonal() << b_acc_cov, b_acc_cov, b_acc_cov;
	// MTK::get_cov<process_noise_input>::type cov = MTK::get_cov<process_noise_input>::type::Zero();
	// MTK::setDiagonal<process_noise_input, vect3, 0>(cov, &process_noise_input::ng, gyr_cov_input);// 0.03
	// MTK::setDiagonal<process_noise_input, vect3, 3>(cov, &process_noise_input::na, acc_cov_input); // *dt 0.01 0.01 * dt * dt 0.05
	// MTK::setDiagonal<process_noise_input, vect3, 6>(cov, &process_noise_input::nbg, b_gyr_cov); // *dt 0.00001 0.00001 * dt *dt 0.3 //0.001 0.0001 0.01
	// MTK::setDiagonal<process_noise_input, vect3, 9>(cov, &process_noise_input::nba, b_acc_cov);   //0.001 0.05 0.0001/out 0.01
	return cov;
}

/* DIFF-DRIVE MODEL: process noise of the diff-drive state (note, eq. (2)/(19)).
 *
 * Noise only drives the random-walk states:
 *   a_dot     = w_a                       -> Q(acc, acc)     = dd_acc_cov
 *   omega_dot = w_omega, anisotropic      -> Q(omg, omg)     = diag(xy, xy, z)
 *   bg_dot    = n_bg, ba_dot = n_ba       -> Q(bg), Q(ba)    = b_gyr_cov, b_acc_cov
 * Optionally a small noise on v (dd_vel_cov, default 0) can absorb unmodelled
 * effects (slip). The note's hard model has none.
 *
 * Convention: esekf::predict adds Q * dt^2 (as in the original Point-LIO), so Q
 * is expressed in the same units and magnitudes as the original
 * acc_cov_output / gyr_cov_output parameters. Q is given directly in error
 * state coordinates, i.e. F_w Q F_w^T of eq. (19) is already applied. */
Eigen::Matrix<double, DD_DIM_STATE, DD_DIM_STATE> process_noise_cov_output()
{
	Eigen::Matrix<double, DD_DIM_STATE, DD_DIM_STATE> cov;
	cov.setZero();
	cov(DD_IDX_VEL, DD_IDX_VEL) = dd_vel_cov;
	cov(DD_IDX_ACC, DD_IDX_ACC) = dd_acc_cov;
	cov.block<3, 3>(DD_IDX_OMG, DD_IDX_OMG).diagonal() << dd_gyr_cov_xy, dd_gyr_cov_xy, dd_gyr_cov_z;
	cov.block<3, 3>(DD_IDX_BG, DD_IDX_BG).diagonal() << b_gyr_cov, b_gyr_cov, b_gyr_cov;
	cov.block<3, 3>(DD_IDX_BA, DD_IDX_BA).diagonal() << b_acc_cov, b_acc_cov, b_acc_cov;
	return cov;
}

Eigen::Matrix<double, 24, 1> get_f_input(state_input &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);
	vect3 a_inertial = s.rot * (in.acc-s.ba); // .normalized()
	for(int i = 0; i < 3; i++ ){
		res(i) = s.vel[i];
		res(i + 3) = omega[i]; 
		res(i + 12) = a_inertial[i] + s.gravity[i]; 
	}
	return res;
}

/* DIFF-DRIVE MODEL: process model f(x, 0), note eq. (2), (5), (6).
 *
 *   R_dot = R [w]x       -> f(rot) = w      (esekf applies R Exp(w dt))
 *   p_dot = R e1 v       -> f(pos) = R e1 v (no lateral / vertical body velocity)
 *   v_dot = a            -> f(vel) = a
 *   a_dot = w_dot = bg_dot = ba_dot = 0 (white noise only)
 *
 * The free-body model had p_dot = G_v, G_v_dot = R I_a + G_g, which lets
 * accelerometer errors integrate into a free vertical / lateral velocity. Here
 * z can only change through  z_dot = e3^T R e1 v,  i.e. only when the robot is
 * pitched AND moving. */
Eigen::Matrix<double, DD_DIM_STATE, 1> get_f_output(state_output &s, const input_ikfom &in)
{
	Eigen::Matrix<double, DD_DIM_STATE, 1> res = Eigen::Matrix<double, DD_DIM_STATE, 1>::Zero();
	const V3D p_dot = s.rot * e1_ * s.vel(0);
	res.block<3, 1>(DD_IDX_POS, 0) = p_dot;
	res.block<3, 1>(DD_IDX_ROT, 0) = s.omg;
	res(DD_IDX_VEL) = s.acc(0);
	return res;
}

Eigen::Matrix<double, 24, 24> df_dx_input(state_input &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 24> cov = Eigen::Matrix<double, 24, 24>::Zero();
	cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
	vect3 acc_;
	in.acc.boxminus(acc_, s.ba);
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);
	cov.template block<3, 3>(12, 3) = -s.rot*MTK::hat(acc_); // .normalized().toRotationMatrix()
	cov.template block<3, 3>(12, 18) = -s.rot; //.normalized().toRotationMatrix();
	// Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
	// Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
	// s.S2_Mx(grav_matrix, vec, 21);
	cov.template block<3, 3>(12, 21) = Eigen::Matrix3d::Identity(); // grav_matrix; 
	cov.template block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity(); 
	return cov;
}

/* DIFF-DRIVE MODEL: df/d(dx) of the process model, note eq. (20).
 *
 * Only the R^n part has to be supplied: esekf::predict builds
 *   F_x = I + df/dx * dt
 * and replaces the attitude rows by Exp(-w dt) and A(w dt) * (df/dx row) on
 * its own (IKFoM convention). Non-zero blocks:
 *
 *   d f(pos)/d theta = -v R [e1]x     (p_k+1 = p + R Exp(dtheta) e1 (v+dv) dt)
 *   d f(pos)/d v     =  R e1
 *   d f(rot)/d w     =  I
 *   d f(vel)/d a     =  1
 *
 * "z-coupling in one line": the only path from attitude to position is
 * -v R [e1]x, so position uncertainty in z grows only with (roll/pitch
 * uncertainty) x speed. In the free-body model it grew through a free G_v_z. */
Eigen::Matrix<double, DD_DIM_STATE, DD_DIM_STATE> df_dx_output(state_output &s, const input_ikfom &in)
{
	Eigen::Matrix<double, DD_DIM_STATE, DD_DIM_STATE> cov = Eigen::Matrix<double, DD_DIM_STATE, DD_DIM_STATE>::Zero();
	const M3D R = s.rot;
	cov.template block<3, 3>(DD_IDX_POS, DD_IDX_ROT) = -s.vel(0) * R * skew(e1_);
	cov.template block<3, 1>(DD_IDX_POS, DD_IDX_VEL) = R * e1_;
	cov.template block<3, 3>(DD_IDX_ROT, DD_IDX_OMG) = Eigen::Matrix3d::Identity();
	cov(DD_IDX_VEL, DD_IDX_ACC) = 1.0;
	return cov;
}

void h_model_input(state_input &s, Eigen::Matrix3d cov_p, Eigen::Matrix3d cov_R, esekfom::dyn_share_modified<double> &ekfom_data)
{
	bool match_in_map = false;
	VF(4) pabcd;
	pabcd.setZero();
	normvec->resize(time_seq[k]);
	int effect_num_k = 0;
	for (int j = 0; j < time_seq[k]; j++)
	{
		PointType &point_body_j  = feats_down_body->points[idx+j+1];
		PointType &point_world_j = feats_down_world->points[idx+j+1];
		pointBodyToWorld(&point_body_j, &point_world_j); 
		V3D p_body = pbody_list[idx+j+1];
		double p_norm = p_body.norm();
		V3D p_world;
		p_world << point_world_j.x, point_world_j.y, point_world_j.z;
		{
			auto &points_near = Nearest_Points[idx+j+1];
            ivox_->GetClosestPoint(point_world_j, points_near, NUM_MATCH_POINTS); // 
			if ((points_near.size() < NUM_MATCH_POINTS)) // || pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5) // 5)
			{
				point_selected_surf[idx+j+1] = false;
			}
			else
			{
				point_selected_surf[idx+j+1] = false;
				if (esti_plane(pabcd, points_near, plane_thr)) //(planeValid)
				{
					float pd2 = fabs(pabcd(0) * point_world_j.x + pabcd(1) * point_world_j.y + pabcd(2) * point_world_j.z + pabcd(3));
					// V3D norm_vec;
					// M3D Rpf, pf;
					// pf = crossmat_list[idx+j+1];
					// // pf << SKEW_SYM_MATRX(p_body);
					// Rpf = s.rot * pf;
					// norm_vec << pabcd(0), pabcd(1), pabcd(2);
					// double noise_state = norm_vec.transpose() * (cov_p+Rpf*cov_R*Rpf.transpose())  * norm_vec + sqrt(p_norm) * 0.001;
					// // if (p_norm > match_s * pd2 * pd2)
					// double epsilon = pd2 / sqrt(noise_state);
					// // cout << "check epsilon:" << epsilon << endl;
					// double weight = 1.0; // epsilon / sqrt(epsilon * epsilon+1);
					// if (epsilon > 1.0) 
					// {
					// 	weight = sqrt(2 * epsilon - 1) / epsilon;
					// 	pabcd(0) = weight * pabcd(0);
					// 	pabcd(1) = weight * pabcd(1);
					// 	pabcd(2) = weight * pabcd(2);
					// 	pabcd(3) = weight * pabcd(3);
					// }
					if (p_norm > match_s * pd2 * pd2)
					{
						point_selected_surf[idx+j+1] = true;
						normvec->points[j].x = pabcd(0);
						normvec->points[j].y = pabcd(1);
						normvec->points[j].z = pabcd(2);
						normvec->points[j].intensity = pabcd(3);
						effect_num_k ++;
					}
				}  
			}
		}
	}
	if (effect_num_k == 0) 
	{
		ekfom_data.valid = false;
		return;
	}
	ekfom_data.M_Noise = laser_point_cov;
	ekfom_data.h_x.resize(effect_num_k, 12);
	ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_num_k, 12);
	ekfom_data.z.resize(effect_num_k);
	int m = 0;
	
	for (int j = 0; j < time_seq[k]; j++)
	{
		// ekfom_data.converge = false;
		if(point_selected_surf[idx+j+1])
		{
			V3D norm_vec(normvec->points[j].x, normvec->points[j].y, normvec->points[j].z);
			
			if (extrinsic_est_en)
			{
				V3D p_body = pbody_list[idx+j+1];
				M3D p_crossmat, p_imu_crossmat;
				p_crossmat << SKEW_SYM_MATRX(p_body);
				V3D point_imu = s.offset_R_L_I * p_body + s.offset_T_L_I;
				p_imu_crossmat << SKEW_SYM_MATRX(point_imu);
				V3D C(s.rot.transpose() * norm_vec);
				V3D A(p_imu_crossmat * C);
				V3D B(p_crossmat * s.offset_R_L_I.transpose() * C);
				ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
			}
			else
			{   
				M3D point_crossmat = crossmat_list[idx+j+1];
				V3D C(s.rot.transpose() * norm_vec); // conjugate().normalized()
				V3D A(point_crossmat * C);
				ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
			}
			ekfom_data.z(m) = -norm_vec(0) * feats_down_world->points[idx+j+1].x -norm_vec(1) * feats_down_world->points[idx+j+1].y -norm_vec(2) * feats_down_world->points[idx+j+1].z-normvec->points[j].intensity;
			
			m++;
		}
	}
	effct_feat_num += effect_num_k;
}

void h_model_output(state_output &s, Eigen::Matrix3d cov_p, Eigen::Matrix3d cov_R, esekfom::dyn_share_modified<double> &ekfom_data)
{
	bool match_in_map = false;
	VF(4) pabcd;
	pabcd.setZero();
	normvec->resize(time_seq[k]);
	int effect_num_k = 0;
	for (int j = 0; j < time_seq[k]; j++)
	{
		PointType &point_body_j  = feats_down_body->points[idx+j+1];
		PointType &point_world_j = feats_down_world->points[idx+j+1];
		pointBodyToWorld(&point_body_j, &point_world_j); 
		V3D p_body = pbody_list[idx+j+1];
		double p_norm = p_body.norm();
		V3D p_world;
		p_world << point_world_j.x, point_world_j.y, point_world_j.z;
		{
			auto &points_near = Nearest_Points[idx+j+1];
			
            ivox_->GetClosestPoint(point_world_j, points_near, NUM_MATCH_POINTS); // 
			
			if ((points_near.size() < NUM_MATCH_POINTS)) // || pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5)
			{
				point_selected_surf[idx+j+1] = false;
			}
			else
			{
				point_selected_surf[idx+j+1] = false;
				if (esti_plane(pabcd, points_near, plane_thr)) //(planeValid)
				{
					float pd2 = fabs(pabcd(0) * point_world_j.x + pabcd(1) * point_world_j.y + pabcd(2) * point_world_j.z + pabcd(3));
					// V3D norm_vec;
					// M3D Rpf, pf;
					// pf = crossmat_list[idx+j+1];
					// // pf << SKEW_SYM_MATRX(p_body);
					// Rpf = s.rot * pf;
					// norm_vec << pabcd(0), pabcd(1), pabcd(2);
					// double noise_state = norm_vec.transpose() * (cov_p+Rpf*cov_R*Rpf.transpose())  * norm_vec + sqrt(p_norm) * 0.001;
					// // if (p_norm > match_s * pd2 * pd2)
					// double epsilon = pd2 / sqrt(noise_state);
					// double weight = 1.0; // epsilon / sqrt(epsilon * epsilon+1);
					// if (epsilon > 1.0) 
					// {
					// 	weight = sqrt(2 * epsilon - 1) / epsilon;
					// 	pabcd(0) = weight * pabcd(0);
					// 	pabcd(1) = weight * pabcd(1);
					// 	pabcd(2) = weight * pabcd(2);
					// 	pabcd(3) = weight * pabcd(3);
					// }
					if (p_norm > match_s * pd2 * pd2)
					{
						// point_selected_surf[i] = true;
						point_selected_surf[idx+j+1] = true;
						normvec->points[j].x = pabcd(0);
						normvec->points[j].y = pabcd(1);
						normvec->points[j].z = pabcd(2);
						normvec->points[j].intensity = pabcd(3);
						effect_num_k ++;
					}
				}  
			}
		}
	}
	if (effect_num_k == 0) 
	{
		ekfom_data.valid = false;
		return;
	}
	/* DIFF-DRIVE MODEL: LiDAR point-to-plane measurement, note eq. (14)/(15).
	 *
	 *   0 = u^T ( R (R_BL L_p + t_BL) + p - q )
	 *   H_L = [ u^T ,  -u^T R [B_p]x ]      (error-state order pos, rot)
	 *
	 * B_p = Lidar_R_wrt_Body * L_p + Lidar_T_wrt_Body is precomputed in
	 * laserMapping.cpp (crossmat_list holds [B_p]x). The original 12-column
	 * Jacobian contained the extrinsic states, which no longer exist, so H_L
	 * is 1 x 6; esekf uses only the leading 6 columns of P for this update.
	 * Written as in the original code: -u^T R [B_p]x = ([B_p]x R^T u)^T. */
	ekfom_data.M_Noise = laser_point_cov;
	ekfom_data.h_x.resize(effect_num_k, 6);
	ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_num_k, 6);
	ekfom_data.z.resize(effect_num_k);
	int m = 0;
	for (int j = 0; j < time_seq[k]; j++)
	{
		// ekfom_data.converge = false;
		if(point_selected_surf[idx+j+1])
		{
			V3D norm_vec(normvec->points[j].x, normvec->points[j].y, normvec->points[j].z);
			M3D point_crossmat = crossmat_list[idx+j+1];
			V3D C(s.rot.transpose() * norm_vec); // conjugate().normalized()
			V3D A(point_crossmat * C);
			ekfom_data.h_x.block<1, 6>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A);
			ekfom_data.z(m) = -norm_vec(0) * feats_down_world->points[idx+j+1].x -norm_vec(1) * feats_down_world->points[idx+j+1].y -norm_vec(2) * feats_down_world->points[idx+j+1].z-normvec->points[j].intensity;

			m++;
		}
	}
	effct_feat_num += effect_num_k;
}

/* DIFF-DRIVE MODEL: IMU measurement model, note sec. 4.1, eq. (7)-(13).
 *
 * The IMU sits at lever arm r_BI (in B) with orientation R_BI (I -> B).
 *
 *   gyro : I_w_m = R_BI^T w + b_g + n_g                                   (7)
 *   acc  : I_a_m = R_BI^T ( a e1 + w x (v e1)          <- B_a_B, eq. (3)
 *                           + w x (w x r_BI)            <- centripetal at the lever arm
 *                           - R^T G_g )                 <- gravity (specific force)
 *                  + b_a + n_a                                            (8)
 *   (the tangential term w_dot x r_BI is neglected, w_dot is not a state;
 *    its effect is absorbed by imu_meas_acc_cov or by mounting the IMU close
 *    to the axle midpoint.)
 *
 * Jacobian H_I (rows: gyro, acc; columns in error-state order):
 *   gyro: d/dw = R_BI^T,  d/db_g = I
 *   acc : d/dtheta = -R_BI^T [R^T g]x                                     (10)
 *         d/dv     =  R_BI^T (w x e1) = R_BI^T [0, w_z, -w_y]^T            (11)
 *         d/da     =  R_BI^T e1                                           (12)
 *         d/dw     =  R_BI^T ( -v [e1]x - [w x r_BI]x - [w]x [r_BI]x )     (13)
 *         d/db_a   =  I
 *
 * ekfom_data.z_IMU receives the residual z - h(x_hat), ekfom_data.h_x_IMU
 * the Jacobian. Saturated channels are flagged in satu_check and removed by
 * esekf::update_iterated_dyn_share_IMU. */
void h_model_IMU_output(state_output &s, esekfom::dyn_share_modified<double> &ekfom_data)
{
    std::memset(ekfom_data.satu_check, false, 6);

	const M3D R_IB  = Imu_R_wrt_Base.transpose(); // R_BI^T : B -> I
	const V3D &r_BI = Imu_T_wrt_Base;
	const M3D R     = s.rot;
	const double v  = s.vel(0);
	const double a  = s.acc(0);
	const V3D w     = s.omg;
	const V3D g_B   = R.transpose() * gravity_world;  // gravity expressed in B

	/* DIFF-DRIVE MODEL (stability fix): angular rate used inside the
	 * accelerometer model.
	 * The centripetal term w x (v e1) and the lever-arm term w x (w x r_BI)
	 * are bilinear / quadratic in w. Using the *state* w there (as in eq. (8))
	 * couples the accelerometer to w through a strongly nonlinear Jacobian;
	 * with a large lever arm (tens of cm) a single linearised EKF step can
	 * then lock onto a wrong (w, a) pair and the filter diverges (observed on
	 * a real robot: w_z -> 7 rad/s while standing, a -> -10 m/s^2).
	 * With imu_gyro_in_acc_model = true (default) these terms use the
	 * bias-corrected gyro reading w_g = R_BI (I_w_m - b_g) instead - the
	 * standard lever-arm compensation of strapdown navigation. The accelerometer
	 * model is then linear in (theta, v, a, b_a) and no longer depends on the
	 * state w (H_aw = 0); w is observed by the gyroscope only. */
	bool gyro_sat = false;
	if (check_satu)
	{
		for (int i = 0; i < 3; i++) gyro_sat = gyro_sat || (fabs(angvel_avr(i)) >= 0.99 * satu_gyro);
	}
	const bool use_gyro = dd_imu_gyro_in_acc_model && !gyro_sat;
	const V3D w_acc = use_gyro ? V3D(Imu_R_wrt_Base * (angvel_avr - s.bg)) : w;

	// predicted measurements
	const V3D acc_axle  = a * e1_ + w_acc.cross(v * e1_);       // B_a_B, eq. (3)
	const V3D acc_lever = w_acc.cross(w_acc.cross(r_BI));       // rigid-body transport to the IMU
	const V3D h_gyr = R_IB * w + s.bg;
	const V3D h_acc = R_IB * (acc_axle + acc_lever - g_B) + s.ba;

	ekfom_data.z_IMU.block<3,1>(0, 0) = angvel_avr - h_gyr;
	ekfom_data.z_IMU.block<3,1>(3, 0) = acc_avr * G_m_s2 / acc_norm - h_acc;
    ekfom_data.R_IMU << imu_meas_omg_cov, imu_meas_omg_cov, imu_meas_omg_cov, imu_meas_acc_cov, imu_meas_acc_cov, imu_meas_acc_cov;

	// Jacobian (6 x 17)
	ekfom_data.h_x_IMU.setZero(6, DD_DIM_STATE);
	ekfom_data.h_x_IMU.block<3,3>(0, DD_IDX_OMG) = R_IB;
	ekfom_data.h_x_IMU.block<3,3>(0, DD_IDX_BG)  = Eigen::Matrix3d::Identity();
	ekfom_data.h_x_IMU.block<3,3>(3, DD_IDX_ROT) = -R_IB * skew(g_B);
	ekfom_data.h_x_IMU.block<3,1>(3, DD_IDX_VEL) = R_IB * w_acc.cross(e1_);
	ekfom_data.h_x_IMU.block<3,1>(3, DD_IDX_ACC) = R_IB * e1_;
	if (!use_gyro)
	{
		// eq. (13); only when the state w is used in the accelerometer model
		ekfom_data.h_x_IMU.block<3,3>(3, DD_IDX_OMG) = R_IB * (-v * skew(e1_) - skew(w.cross(r_BI)) - skew(w) * skew(r_BI));
	}
	// (with the gyro reading, the small dependence of w_g on b_g is neglected)
	ekfom_data.h_x_IMU.block<3,3>(3, DD_IDX_BA)  = Eigen::Matrix3d::Identity();
	if(check_satu)
	{
		if(fabs(angvel_avr(0)) >= 0.99 * satu_gyro)
		{
			ekfom_data.satu_check[0] = true; 
			ekfom_data.z_IMU(0) = 0.0;
		}
		
		if(fabs(angvel_avr(1)) >= 0.99 * satu_gyro) 
		{
			ekfom_data.satu_check[1] = true;
			ekfom_data.z_IMU(1) = 0.0;
		}
		
		if(fabs(angvel_avr(2)) >= 0.99 * satu_gyro)
		{
			ekfom_data.satu_check[2] = true;
			ekfom_data.z_IMU(2) = 0.0;
		}
		
		if(fabs(acc_avr(0)) >= 0.99 * satu_acc)
		{
			ekfom_data.satu_check[3] = true;
			ekfom_data.z_IMU(3) = 0.0;
		}

		if(fabs(acc_avr(1)) >= 0.99 * satu_acc) 
		{
			ekfom_data.satu_check[4] = true;
			ekfom_data.z_IMU(4) = 0.0;
		}

		if(fabs(acc_avr(2)) >= 0.99 * satu_acc) 
		{
			ekfom_data.satu_check[5] = true;
			ekfom_data.z_IMU(5) = 0.0;
		}
	}
}

/* DIFF-DRIVE MODEL: wheel-odometry measurement, note sec. 4.3, eq. (16).
 *
 *   z_W = [ v_m ; w_z,m ] = [ v ; e3^T w ] + n_W
 *   H_W = [ 0 .. 1 (at v) .. 0 ; 0 .. 1 (at w_z) .. 0 ]
 *
 * v_m is scaled by wheel_vel_scale (calibration of the wheel radius). If
 * wheel_use_yaw_rate is false (default, the wheel yaw rate is usually less
 * reliable than the gyroscope) only v_m is used. A chi^2 gate (99 %: 6.63 for
 * 1 DoF, 9.21 for 2 DoF) rejects slip when wheel_gate_en is set. */
bool update_wheel_output(double v_meas, double omega_z_meas, bool use_gate)
{
	const double gate_1dof = 6.635, gate_2dof = 9.210;
	const state_output &s = kf_output.x_;
	if (wheel_use_yaw_rate)
	{
		Eigen::Matrix<double, 2, 1> r;
		r << wheel_vel_scale * v_meas - s.vel(0), omega_z_meas - s.omg(2);
		Eigen::Matrix<double, 2, DD_DIM_STATE> H = Eigen::Matrix<double, 2, DD_DIM_STATE>::Zero();
		H(0, DD_IDX_VEL)     = 1.0;
		H(1, DD_IDX_OMG + 2) = 1.0;
		Eigen::Matrix<double, 2, 2> Rn = Eigen::Matrix<double, 2, 2>::Zero();
		Rn(0, 0) = wheel_vel_cov;
		Rn(1, 1) = wheel_omg_cov;
		return kf_output.update_dense<2>(r, H, Rn, (wheel_gate_en && use_gate) ? gate_2dof : 0.0);
	}
	else
	{
		Eigen::Matrix<double, 1, 1> r;
		r << wheel_vel_scale * v_meas - s.vel(0);
		Eigen::Matrix<double, 1, DD_DIM_STATE> H = Eigen::Matrix<double, 1, DD_DIM_STATE>::Zero();
		H(0, DD_IDX_VEL) = 1.0;
		Eigen::Matrix<double, 1, 1> Rn;
		Rn << wheel_vel_cov;
		return kf_output.update_dense<1>(r, H, Rn, (wheel_gate_en && use_gate) ? gate_1dof : 0.0);
	}
}

/* DIFF-DRIVE MODEL: optional planar prior, note sec. 4.4, eq. (17).
 *
 *   0 = [e1^T; e2^T] R^T e3 + n_P,   H_P = [ 0_2x3 , [e1^T; e2^T] [R^T e3]x , 0 ]
 *   (derivation: (R Exp(dtheta))^T e3 ~ R^T e3 + [R^T e3]x dtheta)
 *
 * and optionally a height prior  0 = e3^T p - z0 + n_z,  H_z = [e3^T, 0 ...].
 * Only valid on flat floors; leave disabled if the robot drives on ramps. */
void update_planar_prior_output()
{
	if (planar_prior_en)
	{
		const V3D zb = kf_output.x_.rot.transpose() * e3_;  // world z in body coordinates
		Eigen::Matrix<double, 2, 1> r;
		r << -zb(0), -zb(1);
		Eigen::Matrix<double, 2, DD_DIM_STATE> H = Eigen::Matrix<double, 2, DD_DIM_STATE>::Zero();
		H.block<2, 3>(0, DD_IDX_ROT) = skew(zb).topRows<2>();
		const double sigma = std::sin(planar_prior_std_deg * PI_M / 180.0);
		Eigen::Matrix<double, 2, 2> Rn = Eigen::Matrix<double, 2, 2>::Identity() * sigma * sigma;
		kf_output.update_dense<2>(r, H, Rn);
	}
	if (height_prior_en)
	{
		Eigen::Matrix<double, 1, 1> r;
		r << height_prior_z0 - kf_output.x_.pos(2);
		Eigen::Matrix<double, 1, DD_DIM_STATE> H = Eigen::Matrix<double, 1, DD_DIM_STATE>::Zero();
		H(0, DD_IDX_POS + 2) = 1.0;
		Eigen::Matrix<double, 1, 1> Rn;
		Rn << height_prior_std * height_prior_std;
		kf_output.update_dense<1>(r, H, Rn);
	}
}

// DIFF-DRIVE MODEL: world-frame velocity G_v = R e1 v (no longer a state).
V3D dd_world_velocity(const state_output &s)
{
	return s.rot * e1_ * s.vel(0);
}

// DIFF-DRIVE MODEL: body acceleration of the axle midpoint, note eq. (3).
V3D dd_body_acceleration(const state_output &s)
{
	return s.acc(0) * e1_ + s.omg.cross(s.vel(0) * e1_);
}

void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);

    V3D p_global;
	// DIFF-DRIVE MODEL: the output filter has no extrinsic states any more
	// (extrinsic_est_en is forced to false in output mode), so the online
	// extrinsic branch only exists for the IMU-input filter. The LiDAR is
	// mapped into the state's body frame with Lidar_R/T_wrt_Body, which is
	// base_link for the diff-drive filter and the IMU for the input filter.
	if (extrinsic_est_en && use_imu_as_input)
	{
		p_global = kf_input.x_.rot * (kf_input.x_.offset_R_L_I * p_body + kf_input.x_.offset_T_L_I) + kf_input.x_.pos;
	}
	else
	{
		if (!use_imu_as_input)
		{
			p_global = kf_output.x_.rot * (Lidar_R_wrt_Body * p_body + Lidar_T_wrt_Body) + kf_output.x_.pos; // .normalized()
		}
		else
		{
			p_global = kf_input.x_.rot * (Lidar_R_wrt_Body * p_body + Lidar_T_wrt_Body) + kf_input.x_.pos; // .normalized()
		}
	}

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}