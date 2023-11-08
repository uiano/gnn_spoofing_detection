import copy
import os

import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS, Birch, MeanShift
from tensorflow_gnn import runner
from tqdm import tqdm

from generators.frame_seq_generator import (FrameSeqGenerator,
                                            NormalSpeedGenerator)


class SpoofingClassifier:

    def is_attack(self, m_frames):
        """"
        Args:

        `m_frames`: a numpy array of shape (num_frames, num_features)
        
        """
        return self.test_statistic(m_frames) > self.threshold

    def test_statistic(self, m_frames):
        """"
        Args:

        `m_frames`: a numpy array of shape (num_frames, num_features)
        
        """
        raise NotImplementedError("method to be overridden by subclasses")


class SonSpoofingClassifier(SpoofingClassifier):
    """Based on a SameOrNotClassifier"""

    def __init__(self, same_or_not_classifier, threshold):
        self.same_or_not_classifier = same_or_not_classifier
        self.threshold = threshold

    def get_son_adjacency(self, m_frames):
        num_frames, num_feat = m_frames.shape

        # m_frames[None,...] is 1 x num_frames x num_features
        m1 = np.tile(m_frames[None, ...], reps=(num_frames, 1, 1))
        # m_frames[:, None, :] num_frames x 1 x num_features
        m2 = np.tile(m_frames[:, None, :], reps=(1, num_frames, 1))
        # m1[:,:,None,:] is num_frames x num_frames x 1 x num_features
        m = np.concatenate((m1[:, :, None, :], m2[:, :, None, :]), axis=3)
        # m_in is num_frames^2 x 2 x num_features
        m_in = np.reshape(m, newshape=(-1, 2, num_feat))

        # length = num_frames^2
        v_pred = self.same_or_not_classifier.are_the_same(m_in)

        m_pred = np.reshape(v_pred, newshape=(num_frames, num_frames))
        #print(m_pred.astype(int))
        np.fill_diagonal(m_pred, 0)
        return m_pred


class GnnSonSpoofingClassifier(SonSpoofingClassifier):

    @staticmethod
    def _build_model(
        graph_tensor_spec,
        node_dim=16,
        edge_dim=16,
        message_dim=64,
        next_state_dim=64,
        num_message_passing=3,
        l2_regularization=5e-4,
        dropout_rate=0.5,
    ):
        # Model building with Keras's Functional API starts with an input object
        # (a placeholder for the eventual inputs). Here is how it works for
        # GraphTensors:
        input_graph = tf.keras.layers.Input(type_spec=graph_tensor_spec)

        # IMPORTANT: All TF-GNN modeling code assumes a GraphTensor of shape []
        # in which the graphs of the input batch have been merged to components of
        # one contiguously indexed graph. (There are no edges between components,
        # so no information flows between them.)
        graph = input_graph.merge_batch_to_components()

        # Nodes and edges have one-hot encoded input features. Sending them through
        # a Dense layer effectively does a lookup in a trainable embedding table.
        def set_initial_node_state(node_set, *, node_set_name):
            # Since we only have one node set, we can ignore node_set_name.
            if node_set_name == 'frame':
                return tf.keras.layers.Dense(node_dim)(
                    node_set[tfgnn.HIDDEN_STATE])

        def set_initial_edge_state(edge_set, *, edge_set_name):
            if edge_set_name == 'same':
                return tf.keras.layers.Dense(edge_dim)(
                    edge_set[tfgnn.HIDDEN_STATE])

        # MapFeatures layer receives callbacks as input for each graph piece: node_set
        # edge_set and context. Each callback applies a transformation over the
        # existing features of the respective graph piece while using a Keras
        # Functional API to call new Keras Layers. For more information and examples
        # about the MapFeatures layer please check out its docstring. This call here
        # initializes the hidden states of the edge and node sets.
        graph = tfgnn.keras.layers.MapFeatures(
            node_sets_fn=set_initial_node_state,
            edge_sets_fn=set_initial_edge_state)(graph)

        num_frames = tf.expand_dims(tf.cast(graph.node_sets["frame"].sizes,
                                            dtype=tf.float32),
                                    axis=-1)
        num_edges = tf.expand_dims(tf.cast(graph.edge_sets["same"].sizes,
                                           dtype=tf.float32),
                                   axis=-1)
        graph = graph.replace_features(
            context={
                tfgnn.HIDDEN_STATE: tf.concat([num_frames, num_edges], axis=1)
            })

        # This helper function is just a short-hand for the code below.
        def dense(units, activation="relu"):
            """A Dense layer with regularization (L2 and Dropout)."""
            regularizer = tf.keras.regularizers.l2(l2_regularization)
            return tf.keras.Sequential([
                tf.keras.layers.Dense(units,
                                      activation=activation,
                                      kernel_regularizer=regularizer,
                                      bias_regularizer=regularizer),
                tf.keras.layers.Dropout(dropout_rate)
            ])

        # The GNN core of the model does `num_message_passing` many updates of node
        # states conditioned on their neighbors and the edges connecting to them.
        # More precisely:
        #  - Each edge computes a message by applying a dense layer `message_fn`
        #    to the concatenation of node states of both endpoints (by default)
        #    and the edge's own unchanging feature embedding.
        #  - Messages are summed up at the common TARGET nodes of edges.
        #  - At each node, a dense layer is applied to the concatenation of the old
        #    node state with the summed edge inputs to compute the new node state.
        # Each iteration of the for-loop creates new Keras Layer objects, so each
        # round of updates gets its own trainable variables.
        for i in range(num_message_passing):
            graph = tfgnn.keras.layers.GraphUpdate(node_sets={
                "frame":
                tfgnn.keras.layers.NodeSetUpdate(
                    {
                        "same":
                        tfgnn.keras.layers.SimpleConv(
                            sender_edge_feature=tfgnn.HIDDEN_STATE,
                            message_fn=dense(message_dim),
                            reduce_type="sum",
                            receiver_tag=tfgnn.TARGET)
                    },
                    tfgnn.keras.layers.NextStateFromConcat(
                        dense(next_state_dim)))
            }, )(graph)

        # After the GNN has computed a context-aware representation of the "frames",
        # the model reads out a representation for the graph as a whole by averaging
        # (pooling) node states into the graph context. The context is global to each
        # input graph of the batch, so the first dimension of the result corresponds
        # to the batch dimension of the inputs (same as the labels).
        readout_features = tfgnn.keras.layers.Pool(
            tfgnn.CONTEXT, "mean", node_set_name="frame")(graph)
        # Context  has a hidden-state feature, concatenate the aggregated node vectors
        # with the hidden-state to get the final vector,
        feat = tf.concat([readout_features, graph.context[tfgnn.HIDDEN_STATE]],
                         axis=1)
        # Put a linear classifier on top (not followed by dropout).
        logits = tf.keras.layers.Dense(1)(feat)

        # Build a Keras Model for the transformation from input_graph to logits.
        return tf.keras.Model(inputs=[input_graph], outputs=[logits])

    graph_schema_pbtxt = """
        node_sets {
            key: "frame"
            value {
                description: "Frame in the sequence of frames."
                features {
                    key: "hidden_state"
                    value: {
                    dtype: DT_FLOAT
                    shape { dim { size: 1 } }
                    }
                }
            }
        }

        edge_sets {
            key: "same"
            value {
                description: "Same or not"
                source: "frame"
                target: "frame"
                features {
                    key: "hidden_state"
                    value: {
                        dtype: DT_FLOAT
                        shape { dim { size: 1 } }
                    }
                }
            }
            
        }

        context {
            features {
                key: "attacked"
                value: {
                description: "[LABEL] Attacked or not (0 -> not; 1 -> attacked)."
                dtype: DT_INT32
                }
            }
        }
        """
    graph_schema = tfgnn.parse_schema(graph_schema_pbtxt)
    graph_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

    @staticmethod
    def extract_labels(graph_tensor):
        """
        Extract the class label from the `GraphTensor`.
        Return a pair compatible with the `tf.keras.Model.fit` method.
        """
        context_features = graph_tensor.context.get_features_dict()
        label = context_features.pop('attacked')
        new_graph_tensor = graph_tensor.replace_features(
            context=context_features)
        return new_graph_tensor, label

    def __init__(self,
                 same_or_not_classifier,
                 threshold,
                 node_dim=16,
                 edge_dim=16,
                 message_dim=64,
                 next_state_dim=64,
                 num_message_passing=3,
                 l2_regularization=5e-4,
                 dropout_rate=0.5,
                 load_weight_from=None):
        """

        Args:
            'node_dim' (int, optional): Dimensions of initial states. Defaults to 16.

            'edge_dim' (int, optional): Dimensions of initial states. Defaults to 16.

            'message_dim' (int, optional): Dimensions for message passing. Defaults to 64.

            'next_state_dim' (int, optional): Dimensions for message passing. Defaults to 64.

            'num_message_passing' (int, optional): Number of message passing steps. Defaults to 3.

            'l2_regularization' (_type_, optional): Defaults to 5e-4.

            'dropout_rate' (float, optional): Defaults to 0.5.

            'load_weight_from' (_type_, optional): Path to the weight file. Defaults to None.
        """
        super().__init__(same_or_not_classifier, threshold)
        graph_schema_wo_context = copy.deepcopy(self.graph_schema)
        graph_schema_wo_context.ClearField('context')
        graph_spec_wo_context = tfgnn.create_graph_spec_from_schema_pb(
            graph_schema_wo_context)
        self.model = self._build_model(graph_spec_wo_context,
                                       node_dim=node_dim,
                                       edge_dim=edge_dim,
                                       message_dim=message_dim,
                                       next_state_dim=next_state_dim,
                                       num_message_passing=num_message_passing,
                                       l2_regularization=l2_regularization,
                                       dropout_rate=dropout_rate)
        if load_weight_from is not None:
            graph, label = self.extract_labels(
                tfgnn.random_graph_tensor(self.graph_spec))
            self.model(graph)
            self.model.load_weights(load_weight_from)

    def create_and_save_datasets(self,
                                 t_meas,
                                 m_meas_locs,
                                 l_max_speeds,
                                 ll_endpoint_inds,
                                 l_frame_rates,
                                 frame_separation,
                                 num_frames_range,
                                 num_train=1000,
                                 num_val=200,
                                 num_test=200,
                                 save_files_to=None):
        """ 
        Create and save the tfrecord files for training, validation, and
        testing.

        Args:

            'num_frames_range' (tuple): range of number of frames

            'num_train' (int, optional): number of training data. Defaults to
            1000.

            'num_val' (int, optional): number of validation data. Defaults to
            200.

            'num_test' (int, optional): number of testing data. Defaults to 200.

            'save_files_to' (str, optional): path to save the tfrecord files.
            Defaults to None.
        """

        assert save_files_to is not None

        def create_graph_data(t_power, m_loc_tx, speed_gen, ll_endpoint_inds,
                              frame_rate, num_frames, frame_separation,
                              b_attack):
            """
            Create a graph.

            Args:
                see create_and_save_datasets

            Returns:
                graph: a GraphTensor
            """
            fsg = FrameSeqGenerator(t_power=t_power,
                                    m_loc_tx=m_loc_tx,
                                    speed=speed_gen,
                                    ll_endpoint_inds=ll_endpoint_inds,
                                    frame_rate=frame_rate,
                                    num_frames=num_frames,
                                    frame_separation=frame_separation)
            m_frames = fsg.gen_observed_frame(b_attack=b_attack)
            m_pred = self.get_son_adjacency(m_frames)
            graph = self.create_graph(adjacency_matrix=m_pred,
                                      context=True,
                                      b_attack=b_attack)
            return graph

        for type_data in ["train", "val", "test"]:
            print(f"Creating {type_data} data...")
            os.makedirs(save_files_to, exist_ok=True)
            with tf.io.TFRecordWriter(save_files_to +
                                      f'{type_data}.tfrecord') as writer:
                if type_data == "train":
                    num_data = num_train
                elif type_data == "val":
                    num_data = num_val
                elif type_data == "test":
                    num_data = num_test
                for _ in tqdm(range(num_data)):
                    num_frames = np.random.randint(num_frames_range[0],
                                                   num_frames_range[1])
                    b_attack = np.random.choice([True, False])
                    # Choose random max speed and frame rate from the given lists
                    speed = np.random.choice(l_max_speeds)
                    speed_gen = NormalSpeedGenerator(mean_speed=speed,
                                                     std_speed=0.1)
                    frame_rate = np.random.choice(l_frame_rates)
                    graph = create_graph_data(t_meas, m_meas_locs, speed_gen,
                                              ll_endpoint_inds, frame_rate,
                                              num_frames, frame_separation,
                                              b_attack)

                    example = tfgnn.write_example(graph)
                    writer.write(example.SerializeToString())

    def load_dataset_and_train(self,
                               load_dataset_from,
                               batch_size=32,
                               learning_rate=1e-3,
                               epochs=200,
                               save_weight_to=None):
        assert load_dataset_from is not None
        for type_data in ["train", "val"]:
            train_dataset_provider = runner.TFRecordDatasetProvider(
                file_pattern=load_dataset_from + f'{type_data}.tfrecord')
            dataset = train_dataset_provider.get_dataset(
                context=tf.distribute.InputContext())
            dataset = dataset.map(
                lambda serialized: tfgnn.parse_single_example(
                    serialized=serialized, spec=self.graph_spec))
            ## Assert that dataset is matched with the spec.
            assert self.graph_spec.is_compatible_with(next(iter(dataset)))
            # Extract the labels from the graph tensors.
            if type_data == "train":
                train_ds = dataset.map(self.extract_labels)
            elif type_data == "val":
                val_ds = dataset.map(self.extract_labels)

        ## Define the loss and metrics
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metrics = [
            tf.keras.metrics.BinaryAccuracy(threshold=self.threshold),
            tf.keras.metrics.BinaryCrossentropy(from_logits=True)
        ]
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer, loss=loss, metrics=metrics)
        self.model.summary()
        batched_train_dataset = train_ds.batch(batch_size)
        batched_val_dataset = val_ds.batch(batch_size)
        ## Callbacks
        call_backs = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             patience=20,
                                             restore_best_weights=True)
        ]

        history = self.model.fit(batched_train_dataset,
                                 epochs=epochs,
                                 validation_data=batched_val_dataset,
                                 callbacks=call_backs)
        if save_weight_to is not None:
            os.makedirs(save_weight_to, exist_ok=True)
            self.model.save_weights(save_weight_to + "gnn_sd.h5")
        return history

    def test_statistic(self, m_frames):
        m_pred = self.get_son_adjacency(m_frames)
        if np.all(m_pred == 0):
            return -1.
        graph = self.create_graph(adjacency_matrix=m_pred, context=False)
        return (self.model(graph)).numpy()[0, 0]

    @staticmethod
    def create_graph(adjacency_matrix, context=True, b_attack=None):
        link = np.array(np.where(adjacency_matrix == 1))
        if context:
            assert b_attack is not None
            graph = tfgnn.GraphTensor.from_pieces(
                node_sets={
                    "frame":
                    tfgnn.NodeSet.from_fields(
                        sizes=tf.constant([adjacency_matrix.shape[0]]),
                        features={
                            "hidden_state":
                            tf.constant(
                                [[ind]
                                 for ind in range(adjacency_matrix.shape[0])],
                                dtype=tf.float32),
                        })
                },
                edge_sets={
                    "same":
                    tfgnn.EdgeSet.from_fields(
                        sizes=tf.constant([link.shape[1]]),
                        adjacency=tfgnn.Adjacency.from_indices(
                            source=("frame",
                                    tf.constant(list(link[0]),
                                                dtype=tf.int32)),
                            target=("frame",
                                    tf.constant(list(link[1]),
                                                dtype=tf.int32))),
                        features={
                            "hidden_state":
                            tf.constant([[1] for _ in range(link.shape[1])],
                                        dtype=tf.float32),
                        }),
                },
                context=tfgnn.Context.from_fields(
                    features={'attacked': [int(b_attack)]}),
            )
        else:
            graph = tfgnn.GraphTensor.from_pieces(
                node_sets={
                    "frame":
                    tfgnn.NodeSet.from_fields(
                        sizes=tf.constant([adjacency_matrix.shape[0]]),
                        features={
                            "hidden_state":
                            tf.constant(
                                [[ind]
                                 for ind in range(adjacency_matrix.shape[0])],
                                dtype=tf.float32),
                        })
                },
                edge_sets={
                    "same":
                    tfgnn.EdgeSet.from_fields(
                        sizes=tf.constant([link.shape[1]]),
                        adjacency=tfgnn.Adjacency.from_indices(
                            source=("frame",
                                    tf.constant(list(link[0]),
                                                dtype=tf.int32)),
                            target=("frame",
                                    tf.constant(list(link[1]),
                                                dtype=tf.int32))),
                        features={
                            "hidden_state":
                            tf.constant([[1] for _ in range(link.shape[1])],
                                        dtype=tf.float32),
                        }),
                })
        return graph

    def __str__(self):
        return "GSD"


class ClusteringSpoofingClassifier(SonSpoofingClassifier):

    def __init__(self, cluster_method, threshold=1.5):
        """
        Args:
            'cluster_method': clustering method to use (DBSCAN, HDBSCAN, OPTICS,
            Birch)

            'threshold': see parent class
        """
        super().__init__(None, threshold)
        self.cluster_method = cluster_method

    def test_statistic(self, m_frames):

        if self.cluster_method == "DBSCAN":
            clustering = DBSCAN(eps=3, min_samples=2).fit(m_frames)
        elif self.cluster_method == "HDBSCAN":
            clustering = HDBSCAN(min_cluster_size=2).fit(m_frames)
        elif self.cluster_method == "OPTICS":
            clustering = OPTICS(min_samples=2, max_eps=3).fit(m_frames)
        elif self.cluster_method == "Birch":
            clustering = Birch(n_clusters=None).fit(m_frames)
        else:
            raise NotImplementedError("Clustering method not implemented")
        labels = clustering.labels_
        # Decision rule
        num_clusters = len(set(labels))
        return num_clusters

    def __str__(self):
        if self.cluster_method == "DBSCAN":
            return "DBSCAN"
        elif self.cluster_method == "HDBSCAN":
            return "HDBSCAN"
        elif self.cluster_method == "OPTICS":
            return "OPTICS"
        elif self.cluster_method == "Birch":
            return "BIRCH"
        else:
            raise NotImplementedError("Clustering method not implemented")
