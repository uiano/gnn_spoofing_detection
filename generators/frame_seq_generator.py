import random

import numpy as np
import scipy


class SpeedGenerator:
    pass


class UniformSpeedGenerator(SpeedGenerator):

    def __init__(self, max_speed):
        self.max_speed = max_speed

    def gen_speed(self):
        return np.random.uniform(0, self.max_speed, 1)

    def __str__(self):
        return f"Speed ~ U(0,{self.max_speed})"


class NormalSpeedGenerator(SpeedGenerator):

    def __init__(self, mean_speed, std_speed):
        self.mean_speed = mean_speed
        self.std_speed = std_speed

    def gen_speed(self):
        return np.abs(np.random.normal(self.mean_speed, self.std_speed, 1))

    def __str__(self):
        return f"Speed ~ N({self.mean_speed},{self.std_speed})"


class FrameSeqGenerator:
    """
    Args:

    - `speed`: a `SpeedGenerator` object or a number. If it is a number, the
      speed is generated as a uniform random variable between 0 and `speed`.

    - `num_frame_noise`: the actual number of frames is drawn uniformly at
      random from the interval [num_frames - num_frame_noise, num_frames +
        num_frame_noise]. This is to see if we can get rid of the wiggly shape
        of the Pd curves.  
    
    """

    def __init__(self,
                 t_power,
                 m_loc_tx,
                 speed,
                 ll_endpoint_inds,
                 frame_rate,
                 num_frames,
                 num_frame_noise=0,
                 frame_separation=1):

        self.t_power = t_power
        self.m_loc_tx = m_loc_tx

        if isinstance(speed, SpeedGenerator):
            self.speed_generator = speed
        else:
            self.speed_generator = UniformSpeedGenerator(speed)

        self.ll_endpoint_inds = ll_endpoint_inds
        self.frame_rate = frame_rate
        self.num_frames = num_frames
        self.num_frame_noise = num_frame_noise
        self.frame_separation = frame_separation

    def gen_observed_frame(self, b_attack):
        """
        If `b_attack`, it generates a sequence of frames (feature vectors)
        that randomly combines the frames of two users moving on different
        lines. 
        
        """

        def gen_random_trajectory(ind_line, frame_rate, num_frames):

            def get_point_in_line(ind_line, dist):
                """
                Returns the point on line `ind_line` that is `dist` away from one of
                the endpoints. 
                """
                ind_start = self.ll_endpoint_inds[ind_line][0]
                ind_end = self.ll_endpoint_inds[ind_line][1]
                loc_start = self.m_loc_tx[ind_start, :]
                loc_end = self.m_loc_tx[ind_end, :]
                v_loc_diff = loc_end - loc_start
                v_loc_diff = v_loc_diff / np.linalg.norm(v_loc_diff)
                return loc_start + dist * v_loc_diff

            def get_line_len(ind_line):
                ind_start = self.ll_endpoint_inds[ind_line][0]
                ind_end = self.ll_endpoint_inds[ind_line][1]
                loc_start = self.m_loc_tx[ind_start, :]
                loc_end = self.m_loc_tx[ind_end, :]
                return np.linalg.norm(loc_end - loc_start)

            def get_trajectory(ind_line, dist_offset, dist_traveled,
                               num_frames):
                """
                Returns a list of points with a trajectory that starts at
                `dist_offset` away from the start of the line, and travels
                `dist_traveled` along the line. The trajectory is sampled at
                `num_frames` points. 
                """
                v_dist = dist_offset + np.linspace(0, dist_traveled,
                                                   num_frames + 1)[:-1]
                v_loc = np.array(
                    [get_point_in_line(ind_line, dist) for dist in v_dist])
                return v_loc

            #speed = np.random.uniform(0, max_speed, 1)
            speed = self.speed_generator.gen_speed()
            tx_time = num_frames / frame_rate
            dist_traveled = tx_time * speed
            dist_offset = np.random.uniform(
                0,
                get_line_len(ind_line) - dist_traveled, 1)

            trajectory = get_trajectory(ind_line, dist_offset, dist_traveled,
                                        num_frames)

            direction = random.sample([-1, 1], 1)[0]
            if direction == 1:
                return trajectory
            else:
                return trajectory[::-1]

        def get_user_frames(m_trajectory, num_frames):
            """
            Returns a matrix with one row per frame, and one column per feature.
            """
            # Compute the distance matrix from each point in m_loc to each point in m_loc_tx
            m_dist = scipy.spatial.distance_matrix(m_trajectory, self.m_loc_tx)
            v_inds_nearest_tx_locs = np.argmin(m_dist, axis=1)
            time_offset = np.random.randint(
                0, self.t_power.shape[2] - self.frame_separation * num_frames,
                1)[0]
            v_time_inds = np.arange(
                time_offset, time_offset + self.frame_separation * num_frames,
                self.frame_separation)
            # num_frames x num_feat
            return np.array([
                self.t_power[v_inds_nearest_tx_locs[ind_frame], :,
                             v_time_inds[ind_frame]]
                for ind_frame in range(num_frames)
            ])

        ind_line_user_1, ind_line_user_2 = random.sample(
            range(len(self.ll_endpoint_inds)), 2)
        num_frames = self.num_frames + np.random.randint(
            -self.num_frame_noise, self.num_frame_noise + 1, 1)[0]

        m_traj_user_1 = gen_random_trajectory(ind_line_user_1, self.frame_rate,
                                              num_frames)
        m_frame_user_1 = get_user_frames(m_traj_user_1, num_frames)

        if b_attack:
            m_traj_user_2 = gen_random_trajectory(ind_line_user_2,
                                                  self.frame_rate, num_frames)
            m_frame_user_2 = get_user_frames(m_traj_user_2, num_frames)

            v_selection_indicator = np.random.randint(0, 2,
                                                      num_frames).astype(bool)
            m_frame = np.where(v_selection_indicator[:, None], m_frame_user_1,
                               m_frame_user_2)
        else:
            m_frame = m_frame_user_1
        return m_frame
