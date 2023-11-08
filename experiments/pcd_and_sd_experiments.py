import pickle

import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence
from tqdm import tqdm

import gsim
from classifiers.same_or_not_classifiers import DnnSameOrNotClassifier
from classifiers.spoofing_classifiers import (ClusteringSpoofingClassifier,
                                              GnnSonSpoofingClassifier)
from generators.frame_seq_generator import (FrameSeqGenerator,
                                            NormalSpeedGenerator)
from generators.indoor_data_generator import DataGenerator
from gsim.gfigure import GFigure
from simulators.spoofing_simulator import SpoofingSimulator

gsim.rs = RandomState(MT19937(SeedSequence(123456789)))
"""

Experiments for position-change detection and graph neural netowk based spoofing
detection (GSD).

Figs: 
- Exp.1001: Training and testing locations
- Exp.1002: RSS vs. y-axis for a fixed x-axis value
- Exp.1003: Region sequences
- Exp.2003: GSD training history
- Exp.3001: ROC curves
- Exp.3002: Pd vs. number of sppeds for a fixed pfa
- Exp.3003: Pd vs. number of frames for a fixed pfa
- Exp.3004: Pd vs. number of samples for a fixed pfa
"""


class ExperimentSet(gsim.AbstractExperimentSet):

    ######################################################
    ### Experiments related to data generation
    ######################################################

    # """ Experiment to process, split and save data """
    def experiment_1001(l_args):
        data_generator = DataGenerator()
        ind_floor = [0]
        num_feat_keep = 10
        num_first_feat = 5
        m_meas_locs, m_meas = data_generator.get_data()
        m_meas_locs, m_meas = data_generator.filter_by_floors(
            ind_floor, m_meas_locs, m_meas)
        m_meas = data_generator.keep_most_observed_feat(m_meas, num_feat_keep)

        # Keep the measurements for which at least the first `num_first_feat`
        # features are observed
        v_inds_keep = np.sum(~np.isnan(m_meas[:, :num_first_feat]),
                             axis=1) == num_first_feat
        m_meas = m_meas[v_inds_keep, :num_first_feat]
        m_meas_locs = m_meas_locs[v_inds_keep, :]

        # Split into train and test data
        num_train = int(m_meas.shape[0] * 0.8)
        v_inds = np.random.permutation(m_meas.shape[0])
        m_meas_train = m_meas[v_inds[:num_train], :]
        m_meas_locs_train = m_meas_locs[v_inds[:num_train], :]
        m_meas_test = m_meas[v_inds[num_train:], :]
        m_meas_locs_test = m_meas_locs[v_inds[num_train:], :]

        # Save test data to pickle
        with open("data/indoor_loc_datasets/indoor_data_split.pkl", 'wb') as f:
            pickle.dump(
                {
                    'm_meas_train': m_meas_train,
                    'm_meas_locs_train': m_meas_locs_train,
                    'm_meas_test': m_meas_test,
                    'm_meas_locs_test': m_meas_locs_test
                }, f)

        # Plot the measurement locations
        G_loc_train = GFigure(
            xaxis=m_meas_locs_train[:, 0],
            yaxis=m_meas_locs_train[:, 1],
            xlabel="x [m]",
            ylabel="y [m]",
            styles='.',
        )
        G_loc_test = GFigure(
            xaxis=m_meas_locs_test[:, 0],
            yaxis=m_meas_locs_test[:, 1],
            xlabel="x [m]",
            ylabel="y [m]",
            styles='r.',
        )
        return [G_loc_train, G_loc_test]

    # """ Experiment to generate Fig. 1 in the paper (Plot Rss on x-axis give a
    # fixe y-axis value) """
    def experiment_1002(l_args):
        # Load data
        with open("data/indoor_loc_datasets/indoor_data_split.pkl", 'rb') as f:
            d_data = pickle.load(f)
        m_meas_train = d_data['m_meas_train']
        m_meas_locs_train = d_data['m_meas_locs_train']
        data_generator = DataGenerator()
        v_xlim = [73, 75]
        v_ylim = [10, 50]
        m_meas_locs, m_meas = data_generator.filter_by_loc(
            m_meas_locs_train, m_meas_train, v_xlim, v_ylim)
        # Sorted by y-axis (ascending) and plot RSS vs y-axis
        v_inds_sorted = np.argsort(m_meas_locs[:, 1])
        m_meas = m_meas[v_inds_sorted, :]
        m_meas_locs = m_meas_locs[v_inds_sorted, :]
        G = GFigure(xlabel="y [m]", ylabel="RSS [dB units]")
        for ind_ap in range(m_meas.shape[1] - 2):
            G.add_curve(xaxis=m_meas_locs[:, 1],
                        yaxis=m_meas[:, ind_ap],
                        legend=f"AP {ind_ap}",
                        styles='.-')
        return G

    # """ Examples to generate Fig. 2 in the paper (region sequences) """
    def experiment_1003(l_args):

        def combine_seqs(l_v1, l_v2):
            l_v_out = []
            for v1, v2 in zip(l_v1, l_v2):
                if np.random.uniform(size=1) < 0.5:
                    l_v_out.append(v1)
                else:
                    l_v_out.append(v2)
            return l_v_out

        v_ind_frames = np.arange(30)
        lv_region_seqs = [
            np.round(v_ind_frames / 7),
            combine_seqs(np.zeros(v_ind_frames.shape),
                         np.ones(v_ind_frames.shape)),
            #
            combine_seqs(1 + np.round(v_ind_frames / 7),
                         np.zeros(v_ind_frames.shape)),
            combine_seqs(3 + np.round(v_ind_frames / 7),
                         np.round(v_ind_frames / 10)),
        ]

        G = GFigure(num_subplot_columns=2)
        for v_region_seq, title in zip(lv_region_seqs,
                                       ["(a)", "(b)", "(c)", "(d)"]):
            G.next_subplot(yaxis=v_region_seq,
                           styles='.-',
                           mode="stem",
                           xlabel="Frame index",
                           ylabel="Region index",
                           title=title)

        return G

    ######################################################
    ### Experiments related to PCD and GNN training
    ######################################################
    # """ Experiment to train the PCD. Pre-trained weights are provided in the
    # repo. """
    def experiment_2001(l_args):
        # Load datas
        with open("data/indoor_loc_datasets/indoor_data_split.pkl", 'rb') as f:
            d_data = pickle.load(f)
        data_generator = DataGenerator()
        t_meas_train = data_generator.generate_samples(
            m_meas=d_data['m_meas_train'],
            num_samples=16,
            num_feat_realizations=1000)

        # Instantiate and train the PCD
        sonc = DnnSameOrNotClassifier(num_epochs=350,
                                      verbosity=1,
                                      run_eagerly=False)
        sonc.train(t_meas_train,
                   num_pairs_train=10000,
                   num_pairs_val=2000,
                   val_split=0.2)

        # Save weights
        sonc.save(path="output/dnn_weights/son_indoor.h5")

    # """ Experiments for creating and saving datasets for GNN spoofing detection.
    # Pre-saved datasets are provided in the repo. """
    def experiment_2002(l_args):
        # Load data
        with open("data/indoor_loc_datasets/indoor_data_split.pkl", 'rb') as f:
            d_data = pickle.load(f)
        data_generator = DataGenerator()
        m_meas_train = d_data['m_meas_train']
        m_meas_locs_train = d_data['m_meas_locs_train']
        _, ll_endpont_inds_train = data_generator.get_endpoints(
            m_meas_locs=m_meas_locs_train,
            min_points_per_line=12,
            min_line_len=35,
            dist_threshold=6.)
        l_max_speeds = [0, 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8,
                        2.0]  # m/s
        l_frame_rates = [10]  # frames/second
        num_frames_range = (3, 60)
        frame_separation = 1

        # Generate samples
        t_meas_train = data_generator.generate_samples(
            m_meas=m_meas_train, num_samples=16, num_feat_realizations=1000)

        # Instantiate and load the PCD
        sonc = DnnSameOrNotClassifier(num_epochs=350,
                                      verbosity=1,
                                      run_eagerly=False)
        sonc.load(path="output/dnn_weights/son_indoor.h5",
                  num_feat=t_meas_train.shape[1])

        # Instantiate the GNN spoofing detector (GSD)
        gnnsc = GnnSonSpoofingClassifier(same_or_not_classifier=sonc,
                                         threshold=0.)

        # Create and save datasets
        gnnsc.create_and_save_datasets(
            t_meas=t_meas_train,
            m_meas_locs=m_meas_locs_train,
            l_max_speeds=l_max_speeds,
            ll_endpoint_inds=ll_endpont_inds_train,
            l_frame_rates=l_frame_rates,
            frame_separation=frame_separation,
            num_frames_range=num_frames_range,
            num_train=20000,
            num_val=4000,
            num_test=4000,
            save_files_to='data/indoor_loc_datasets/gnn_dataset/')

    # """ Experiments for loading dataset and training GNN spoofing detection.
    # Pre-trained weights are provided in the repo."""
    def experiment_2003(l_args):
        # Instantiate the GNN spoofing detector (GSD) and train
        gnnsc = GnnSonSpoofingClassifier(same_or_not_classifier=None,
                                         threshold=0.)
        history = gnnsc.load_dataset_and_train(
            load_dataset_from='data/indoor_loc_datasets/gnn_dataset/',
            batch_size=100,
            learning_rate=1e-3,
            epochs=400,
            save_weight_to='output/gnn_weights/')

        # Plot training history
        G_loss = GFigure(xaxis=history.epoch,
                         yaxis=history.history['loss'],
                         xlabel="Epoch",
                         ylabel="Loss",
                         grid=True)
        G_loss.add_curve(xaxis=history.epoch,
                         yaxis=history.history['val_loss'],
                         legend="Validation loss")
        G_acc = GFigure(xaxis=history.epoch,
                        yaxis=history.history['binary_accuracy'],
                        xlabel="Epoch",
                        ylabel="Accuracy",
                        grid=True)
        G_acc.add_curve(xaxis=history.epoch,
                        yaxis=history.history['val_binary_accuracy'],
                        legend="Validation accuracy")
        return [G_loss, G_acc]

    ######################################################
    ### Experiments for performance evaluation
    ######################################################
    # """ Examples to generate Fig. 3 in the paper (ROC curves) """
    def experiment_3001(l_args):
        # Load data
        with open("data/indoor_loc_datasets/indoor_data_split.pkl", 'rb') as f:
            d_data = pickle.load(f)
        data_generator = DataGenerator()
        m_meas_test = d_data['m_meas_test']
        m_meas_locs_test = d_data['m_meas_locs_test']
        _, ll_endpont_inds_test = data_generator.get_endpoints(
            m_meas_locs=m_meas_locs_test,
            min_points_per_line=12,
            min_line_len=35,
            dist_threshold=6.)
        max_speed = 2  # m/s
        frame_rate = 10  # frames/second
        num_frames = 30
        frame_separation = 1
        num_mc_realizations = 10000
        # Generate samples
        t_meas_test = data_generator.generate_samples(
            m_meas=m_meas_test, num_samples=150, num_feat_realizations=1000)

        # Instantiate and load the PCD
        sonc = DnnSameOrNotClassifier(num_epochs=350,
                                      verbosity=1,
                                      run_eagerly=False)
        sonc.load(path="output/dnn_weights/son_indoor.h5",
                  num_feat=t_meas_test.shape[1])

        # Instantiate and load the GNN spoofing detector (GSD)
        gnnsc = GnnSonSpoofingClassifier(
            same_or_not_classifier=sonc,
            threshold=0.,
            load_weight_from='output/gnn_weights/gnn_sd.h5')

        # List of detectors to compare
        l_classifiers = [
            gnnsc,
            ClusteringSpoofingClassifier(cluster_method='DBSCAN'),
            ClusteringSpoofingClassifier(cluster_method='HDBSCAN'),
            ClusteringSpoofingClassifier(cluster_method='OPTICS'),
            ClusteringSpoofingClassifier(cluster_method='Birch'),
        ]

        fsg = FrameSeqGenerator(t_power=t_meas_test,
                                m_loc_tx=m_meas_locs_test,
                                speed=max_speed,
                                ll_endpoint_inds=ll_endpont_inds_test,
                                frame_rate=frame_rate,
                                num_frames=num_frames,
                                frame_separation=frame_separation)
        # First column contains realizations under H0 and second column under H1
        G_roc = GFigure(xlabel="Pfa", ylabel="Pd", grid=True)
        for classifier in l_classifiers:
            m_stat_vals = np.zeros((num_mc_realizations, 2))
            for ind, b_attack in enumerate([False, True]):
                m_stat_vals[:,
                            ind] = SpoofingSimulator.get_test_statistic_realizations(
                                fsg,
                                classifier,
                                num_mc_realizations,
                                b_attack=b_attack)

            # Plot ROC
            l_thresholds = sorted(list(set(np.ravel(m_stat_vals))))
            # First column is Pfa, second Pd
            m_pfa_pd = np.array([
                (np.sum(m_stat_vals[:, 0] >= threshold) / num_mc_realizations,
                 np.sum(m_stat_vals[:, 1] >= threshold) / num_mc_realizations)
                for threshold in l_thresholds
            ])
            G_roc.add_curve(xaxis=m_pfa_pd[:, 0],
                            yaxis=m_pfa_pd[:, 1],
                            legend=str(classifier))

        return G_roc

    # """Examples to generate Fig. 4 in the paper (Pd vs number of sppeds for a
    # fixed pfa)"""
    def experiment_3002(l_args):
        # Load data
        with open("data/indoor_loc_datasets/indoor_data_split.pkl", 'rb') as f:
            d_data = pickle.load(f)
        data_generator = DataGenerator()
        m_meas_test = d_data['m_meas_test']
        m_meas_locs_test = d_data['m_meas_locs_test']
        _, ll_endpont_inds_test = data_generator.get_endpoints(
            m_meas_locs=m_meas_locs_test,
            min_points_per_line=12,
            min_line_len=35,
            dist_threshold=6.)
        l_speeds = [0, 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2.0]  # m/s
        frame_rate = 10  # frames/second
        frame_separation = 1
        num_frames = 30
        num_realizations = 10000
        pfa = 0.1
        # Generate samples
        t_meas_test = data_generator.generate_samples(
            m_meas=m_meas_test, num_samples=150, num_feat_realizations=1000)

        # Instantiate and load the PCD
        sonc = DnnSameOrNotClassifier()
        sonc.load(path="output/dnn_weights/son_indoor.h5",
                  num_feat=t_meas_test.shape[1])

        # Instantiate and load the GNN spoofing detector (GSD)
        gnnsc = GnnSonSpoofingClassifier(
            same_or_not_classifier=sonc,
            threshold=0.,
            load_weight_from='output/gnn_weights/gnn_sd.h5')

        # List of detectors to compare
        l_classifiers = [
            gnnsc,
            ClusteringSpoofingClassifier(cluster_method='DBSCAN'),
            ClusteringSpoofingClassifier(cluster_method='HDBSCAN'),
            ClusteringSpoofingClassifier(cluster_method='OPTICS'),
            ClusteringSpoofingClassifier(cluster_method='Birch'),
        ]

        # Evaluation and plot
        G = GFigure(xlabel="Speed (m/s)", ylabel="Pd", grid=True)

        for classifier in tqdm(l_classifiers):
            l_speed_gen = [
                NormalSpeedGenerator(mean_speed=speed, std_speed=0.01)
                for speed in l_speeds
            ]
            l_fsg = [
                FrameSeqGenerator(t_power=t_meas_test,
                                  m_loc_tx=m_meas_locs_test,
                                  speed=speed_gen,
                                  ll_endpoint_inds=ll_endpont_inds_test,
                                  frame_rate=frame_rate,
                                  num_frames=num_frames,
                                  frame_separation=frame_separation)
                for speed_gen in l_speed_gen
            ]
            l_pd = SpoofingSimulator.get_pd_given_pfa(l_fsg, classifier,
                                                      num_realizations, pfa)

            G.add_curve(xaxis=l_speeds, yaxis=l_pd, legend=str(classifier))
        return G

    # """ Examples to generate Fig. 5 in the paper (Pd vs number of frames for a
    # fixed pfa) """
    def experiment_3003(l_args):
        # Load data
        with open("data/indoor_loc_datasets/indoor_data_split.pkl", 'rb') as f:
            d_data = pickle.load(f)
        data_generator = DataGenerator()
        m_meas_test = d_data['m_meas_test']
        m_meas_locs_test = d_data['m_meas_locs_test']
        _, ll_endpont_inds_test = data_generator.get_endpoints(
            m_meas_locs=m_meas_locs_test,
            min_points_per_line=12,
            min_line_len=35,
            dist_threshold=6.)
        max_speed = 2  # m/s
        speed_gen = NormalSpeedGenerator(mean_speed=max_speed, std_speed=0.01)
        frame_rate = 10  # frames/second
        frame_separation = 1
        l_num_frames = [
            3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80
        ]
        num_realizations = 10000
        pfa = 0.1
        # Generate samples
        t_meas_test = data_generator.generate_samples(
            m_meas=m_meas_test, num_samples=150, num_feat_realizations=1000)

        # Instantiate and load the PCD
        sonc = DnnSameOrNotClassifier()
        sonc.load(path="output/dnn_weights/son_indoor.h5",
                  num_feat=t_meas_test.shape[1])

        # Instantiate and load the GNN spoofing detector (GSD)
        gnnsc = GnnSonSpoofingClassifier(
            same_or_not_classifier=sonc,
            threshold=0.,
            load_weight_from='output/gnn_weights/gnn_sd.h5')

        # List of detectors to compare
        l_classifiers = [
            gnnsc,
            ClusteringSpoofingClassifier(cluster_method='DBSCAN'),
            ClusteringSpoofingClassifier(cluster_method='HDBSCAN'),
            ClusteringSpoofingClassifier(cluster_method='OPTICS'),
            ClusteringSpoofingClassifier(cluster_method='Birch'),
        ]

        # Evaluation and plot
        G = GFigure(xlabel="Number of frames", ylabel="Pd", grid=True)
        for classifier in tqdm(l_classifiers):
            l_pd = []
            for num_frames in tqdm(l_num_frames):
                fsg = FrameSeqGenerator(t_power=t_meas_test,
                                        m_loc_tx=m_meas_locs_test,
                                        speed=speed_gen,
                                        ll_endpoint_inds=ll_endpont_inds_test,
                                        frame_rate=frame_rate,
                                        num_frames=num_frames,
                                        frame_separation=frame_separation)
                l_pd += [
                    SpoofingSimulator.get_pd_given_pfa(fsg, classifier,
                                                       num_realizations, pfa)
                ]
            G.add_curve(xaxis=l_num_frames, yaxis=l_pd, legend=str(classifier))
        return G

    # """Examples to generate Fig. 6 in the paper (Pd vs number of samples for a
    # fixed pfa) """
    def experiment_3004(l_args):
        # Load data
        with open("data/indoor_loc_datasets/indoor_data_split.pkl", 'rb') as f:
            d_data = pickle.load(f)
        data_generator = DataGenerator()
        m_meas_test = d_data['m_meas_test']
        m_meas_locs_test = d_data['m_meas_locs_test']
        _, ll_endpont_inds_test = data_generator.get_endpoints(
            m_meas_locs=m_meas_locs_test,
            min_points_per_line=12,
            min_line_len=35,
            dist_threshold=6.)
        max_speed = 2  # m/s
        speed_gen = NormalSpeedGenerator(mean_speed=max_speed, std_speed=0.01)
        frame_rate = 10  # frames/second
        frame_separation = 1
        num_frames = 30
        l_num_samples = [
            1, 2, 4, 6, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180
        ]
        num_realizations = 10000
        pfa = 0.1

        # Instantiate and load the PCD
        sonc = DnnSameOrNotClassifier()
        sonc.load(path="output/dnn_weights/son_indoor.h5",
                  num_feat=m_meas_test.shape[1])

        # Instantiate and load the GNN spoofing detector (GSD)
        gnnsc = GnnSonSpoofingClassifier(
            same_or_not_classifier=sonc,
            threshold=0.,
            load_weight_from='output/gnn_weights/gnn_sd.h5')

        # List of detectors to compare
        l_classifiers = [
            gnnsc,
            ClusteringSpoofingClassifier(cluster_method='DBSCAN'),
            ClusteringSpoofingClassifier(cluster_method='HDBSCAN'),
            ClusteringSpoofingClassifier(cluster_method='OPTICS'),
            ClusteringSpoofingClassifier(cluster_method='Birch'),
        ]
        # Evaluation and plot
        G = GFigure(xlabel="Number of samples", ylabel="Pd", grid=True)
        for classifier in l_classifiers:
            l_pd = []
            for num_samples in tqdm(l_num_samples):
                # Generate samples
                t_meas_test = data_generator.generate_samples(
                    m_meas=m_meas_test,
                    num_samples=num_samples,
                    num_feat_realizations=1000)
                fsg = FrameSeqGenerator(t_power=t_meas_test,
                                        m_loc_tx=m_meas_locs_test,
                                        speed=speed_gen,
                                        ll_endpoint_inds=ll_endpont_inds_test,
                                        frame_rate=frame_rate,
                                        num_frames=num_frames,
                                        frame_separation=frame_separation)

                l_pd += [
                    SpoofingSimulator.get_pd_given_pfa(fsg, classifier,
                                                       num_realizations, pfa)
                ]
            G.add_curve(xaxis=l_num_samples,
                        yaxis=l_pd,
                        legend=str(classifier))
        return G
