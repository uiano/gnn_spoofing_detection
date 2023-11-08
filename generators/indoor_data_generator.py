import os
import pickle

import numpy as np
import scipy


def get_floors(m_meas_locs):
    """
    Returns:
        list of floors
    """
    return sorted(list(set(m_meas_locs[:, 2])))


class DataGenerator():
    """
    Data generator for dataset (Lohan - https://zenodo.org/record/889798). 

    Data is read from files provied by the authors using function
    'get_data_from_file' and saved to a pickle file named 'indoor_data.pickle'.
    
    One wants to have the raw data, please download it from the link above.

    One can also use another dataset by processing the data and saving it to a 
    pickle file and then providing the path to the pickle file to the constructor.

    """
    threshold = 0.3  # threshold for determining if a point is in a line

    def __init__(self, path_to_data=None):
        if path_to_data is None:
            self.data_file_name = 'data/indoor_loc_datasets/indoor_data.pickle'
        else:
            self.data_file_name = path_to_data

    def get_data(self):
        """

        Returns:
        
            'cooordinates': (num_points x 3) 3D coordinates of measurement
            locations. If the coordinates are 2D, add a zero z-coordinate

            'rss': (num_points x num_aps) received signal strength (RSS) values
            of measurement locations
        """

        if os.path.exists(self.data_file_name):
            with open(self.data_file_name, 'rb') as file:
                coordinates, rss = pickle.load(file)
        else:
            raise FileNotFoundError(
                f'File {self.data_file_name} does not exist')
        # If the coordinates are 2D, add a zero z-coordinate
        if coordinates.shape[1] == 2:
            coordinates = np.concatenate(
                [coordinates, np.zeros((coordinates.shape[0], 1))], axis=1)
        return coordinates, rss

    @staticmethod
    def filter_by_floors(ind_floor, m_meas_locs, m_meas=None):
        """Filter data by floor

        Args:
            'ind_floor' (int or list): indices of floor

            'm_meas_locs': (num_points x 3) 3D coordinates of
            measurement locations

            'm_meas': (num_points x num_aps) received signal strength (RSS)
            values of measurement locations

        Returns:
            'm_meas_locs': (num_points_this_floor x 3) 3D coordinates of
            measurement locations

            'm_meas': (num_points_this_floor x num_aps) received signal strength
            (RSS) values of measurement locations
        """
        if isinstance(ind_floor, int):
            ind_floor = [ind_floor]
        v_floor_heights = get_floors(m_meas_locs)
        v_inds = []
        for ind in ind_floor:
            v_inds_filtered = np.where(
                m_meas_locs[:, 2] == v_floor_heights[ind])[0]
            v_inds.extend(list(v_inds_filtered))
            if m_meas is None:
                return m_meas_locs[v_inds, :]
        return m_meas_locs[v_inds, :], m_meas[v_inds, :]

    @staticmethod
    def filter_by_loc(m_meas_locs, m_meas, v_xlim, v_ylim):
        """
        Returns:

            m_meas_locs_filt: num_meas_filt x 3 matrix with the measurement
            locations in `m_meas_locs` that are in the rectangle defined by
            `v_xlim` and `v_ylim`

            m_meas_filt: num_meas_filt x num_feat matrix with the corresponding
            features. 
        
        """
        v_inds = np.where((m_meas_locs[:, 0] >= v_xlim[0])
                          & (m_meas_locs[:, 0] <= v_xlim[1])
                          & (m_meas_locs[:, 1] >= v_ylim[0])
                          & (m_meas_locs[:, 1] <= v_ylim[1]))[0]
        return m_meas_locs[v_inds, :], m_meas[v_inds, :]

    @staticmethod
    def keep_most_observed_feat(m_meas, num_feat_keep):
        """
        Returns:

            - 'm_meas_keep': (num_points x num_feat_keep) Submatrix of m_meas that
              contains the `num_feat_keep` columns that contain fewest NaNs
        
        """
        v_num_obs = np.sum(~np.isnan(m_meas), axis=0)
        v_ind = np.argsort(v_num_obs)[::-1]
        return m_meas[:, v_ind[:num_feat_keep]]

    def get_endpoints(self,
                      m_meas_locs,
                      min_points_per_line,
                      min_line_len,
                      dist_threshold=5.):
        """Get endpoints of lines

        Args:
            'm_meas_locs': (num_points x 3) 3D coordinates of measurement

            'min_points_per_line': minimum number of points per line

            'min_line_len': minimum length of line

            'dist_threshold': maximum distance between 2 consecutive points in a
            line


        Returns:
            'll_end_points': list of endpoints of lines
            
            'll_end_point_inds': list of indicies of endpoints of lines
        """
        # Find pairs of points that form lines with a length greater than the
        # min_line_len
        ll_end_points = []
        ll_end_point_inds = []
        for ind_floor in range(self.num_floors(m_meas_locs=m_meas_locs)):
            m_meas_locs_this_floor = self.filter_by_floors(
                ind_floor, m_meas_locs)
            m_dists = scipy.spatial.distance_matrix(m_meas_locs_this_floor,
                                                    m_meas_locs_this_floor)
            # Remove the lower triangle
            m_dists[np.tril_indices_from(m_dists)] = 0.
            l_ind_end_points = np.argwhere(m_dists > min_line_len)
            # Keep lines that have at least min_points_per_line points
            for ind_start, ind_end in l_ind_end_points:
                end_point = m_meas_locs_this_floor[ind_end]
                start_point = m_meas_locs_this_floor[ind_start]
                v_inds_in_line = self.get_point_indicies_in_line(
                    [start_point, end_point], m_meas_locs_this_floor)
                if len(v_inds_in_line) > min_points_per_line:
                    # Keep only lines that have points uniformly distributed
                    # along the line
                    v_sorted_dist_to_start = np.array(
                        sorted(
                            self.calculate_projected_point_2_start_point_dist([
                                m_meas_locs_this_floor[ind_start],
                                m_meas_locs_this_floor[ind_end]
                            ], m_meas_locs_this_floor[v_inds_in_line])))
                    if np.all((v_sorted_dist_to_start[1:] -
                               v_sorted_dist_to_start[:-1]) <= dist_threshold):
                        ll_end_points.append([
                            m_meas_locs_this_floor[ind_start],
                            m_meas_locs_this_floor[ind_end]
                        ])
                        ll_end_point_inds.append([ind_start, ind_end])
        return ll_end_points, ll_end_point_inds

    def get_point_indicies_in_line(self, l_endpoints, l_points):
        """Get point indicies in a line

        Args:
            'l_endpoints': list of endpoints of in a line

            'l_points': list of points

        Returns:
            'v_inds_in_line': vector of indicies of points in lines
        """
        # Only use point coordinates in the same floor
        l_points_in_floor = l_points[l_points[:, 2] == l_endpoints[0][2]]
        v_dists = self.calculate_point_2_line_dist(l_endpoints,
                                                   l_points_in_floor)
        v_inds_in_line = np.where(np.abs(v_dists) < self.threshold)[0]
        # Find indices in the original list of points
        v_inds_in_line = np.where(
            l_points[:, 2] == l_endpoints[0][2])[0][v_inds_in_line]
        return v_inds_in_line

    def calculate_projected_point_2_start_point_dist(self, l_endpoints,
                                                     l_points):
        v_dists = self.calculate_point_2_line_dist(l_endpoints, l_points)
        # Calulate the distance of points to the start point
        v_dists_start_point = np.linalg.norm(l_points - l_endpoints[0], axis=1)
        v_projected_dist = np.sqrt(v_dists_start_point**2 - v_dists**2)
        return v_projected_dist

    @staticmethod
    def calculate_point_2_line_dist(l_endpoints, l_points):
        """Calulate the distance of points to a line

        Args:
            'l_endpoints': list of endpoints of in a line

            'l_points': list of points in the same floor as the endpoints

        Returns:
            'v_dists': vector of distances of points to the line
        """
        # Check endpoints and points are in the same floor
        assert l_endpoints[0][2] == l_endpoints[1][2]
        assert l_points[0][2] == l_points[1][2]
        v_line = l_endpoints[1] - l_endpoints[0]
        v_line = v_line / np.linalg.norm(v_line)
        v_perp = np.array([-v_line[1], v_line[0]])
        v_dists = (l_points - l_endpoints[0])[..., :2] @ v_perp
        return v_dists

    @staticmethod
    def generate_samples(m_meas, num_samples, num_feat_realizations=1):
        """
        Args:
            'm_meas': num_points x num_feat matrix with the RSS values

            'num_samples': degrees of freedom of chi-square distribution. Number
            of time samples used to estimate the RSS values

            'num_realizations': number of realizations to generate

        Returns:
            't_meas': num_points x num_feat x num_samples tensor of samples
        """
        # Tranform m_meas to natural unit
        m_meas = 10**(m_meas / 10)
        t_noise = np.random.chisquare(num_samples,
                                      size=(num_feat_realizations,
                                            m_meas.shape[0], m_meas.shape[1]))
        t_meas = m_meas * t_noise / (2 * num_samples)
        t_meas = t_meas.transpose(1, 2, 0)
        t_meas = 10 * np.log10(t_meas)
        return t_meas
