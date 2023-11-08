import numpy as np


class SpoofingSimulator:

    @staticmethod
    def get_prob_metrics(frame_gen, spoofing_classifier, num_mc_iter):
        """
        Returns the probability of error of a spoofing classifier.

        Args:
            frame_gen: a FrameSeqGenerator object
            spoofing_classifier: a SpoofingClassifier object
            num_mc_iter: number of MC iterations

        Returns:
            prob_error: the probability of error of the spoofing classifier
            pfa: the probability of false alarm
            pd: the probability of detection
        """
        v_attack = np.random.choice([True, False], size=num_mc_iter)
        l_attack_pred = []
        for ind_mc_iter in range(num_mc_iter):
            #print(f"attack = {v_attack[ind_mc_iter]}")
            m_frames = frame_gen.gen_observed_frame(v_attack[ind_mc_iter])
            l_attack_pred += [spoofing_classifier.is_attack(m_frames)]

        v_attack_pred = np.array(l_attack_pred)
        prob_err = np.mean(v_attack_pred != v_attack)
        pfa = np.mean(v_attack_pred[v_attack == False])
        pd = np.mean(v_attack_pred[v_attack == True])
        return prob_err, pfa, pd

    @staticmethod
    def get_test_statistic_realizations(l_frame_gens, spoofing_classifier,
                                        num_realizations, b_attack):
        """
        Returns a vector of length `num_realizations` with values of the
        test statistic of the spoofing detector.
        
        """
        l_values = []
        if not isinstance(l_frame_gens, list):
            l_frame_gens = [l_frame_gens]
        for _ in range(num_realizations):
            frame_gen = np.random.choice(l_frame_gens)
            m_frames = frame_gen.gen_observed_frame(b_attack)
            l_values += [spoofing_classifier.test_statistic(m_frames)]

        return l_values

    @staticmethod
    def get_pd_given_pfa(l_frame_gens, spoofing_classifier, num_realizations,
                         pfa):
        """
        Args:
            'l_frame_gens': a list of FrameSeqGenerator objects

            'spoofing_classifier': a SpoofingClassifier object

            'num_realizations': number of MC realizations

            'pfa': the probability of false alarm
        Returns:
            If 'l_frame_gens' is not a list, returns the probability of
            detection given an upper bound on the probability of false alarm.

            If 'l_frame_gens' is a list, returns a list of probabilities of
            detection given an upper bound on the probability of false alarm.

            The threshold is set accordingly. 
        
        """
        v_values_no_attack = np.array(
            SpoofingSimulator.get_test_statistic_realizations(
                l_frame_gens,
                spoofing_classifier,
                num_realizations,
                b_attack=False))
        # The threshold is the smallest number such that mean(v_values > threshold) <= pfa
        threshold = np.percentile(v_values_no_attack, 100 * (1 - pfa))
        if not isinstance(l_frame_gens, list):
            l_values_w_attack = SpoofingSimulator.get_test_statistic_realizations(
                l_frame_gens,
                spoofing_classifier,
                num_realizations,
                b_attack=True)
            return np.mean(l_values_w_attack > threshold)
        else:
            return [
                np.mean(
                    SpoofingSimulator.get_test_statistic_realizations(
                        frame_gen,
                        spoofing_classifier,
                        num_realizations,
                        b_attack=True) > threshold)
                for frame_gen in l_frame_gens
            ]
