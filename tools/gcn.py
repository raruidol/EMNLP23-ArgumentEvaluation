import functools
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph
import flax
import haiku as hk
import optax
import numpy as onp
import networkx as nx
from typing import Tuple
from sentence_transformers import SentenceTransformer

WE_MODEL = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')


def networkx_to_jraph(nx_graph):
    mapping = {}
    i = 0
    for n in nx_graph.nodes():
        mapping[n] = i
        i += 1

    # arbitrary int ids to ordered 0,1,2,... ids
    nx_graph_c = nx.relabel_nodes(nx_graph, mapping, copy=True)
    nl_nodes = [nx_graph_c.nodes[n]['text_cat'] for n in nx_graph_c.nodes()]

    # generate sentence embeddings from the text_cat feature and use it as node features
    node_features = jnp.array(WE_MODEL.encode(nl_nodes))

    # define senders and receivers based on the (conflicting) edges of the nx graph
    ef = []
    snd = []
    rec = []
    attacks = [e for e in nx_graph_c.edges]
    for att in attacks:
        snd.append(att[0])
        rec.append(att[1])
        ef.append([1.])
    senders = jnp.array(snd)
    receivers = jnp.array(rec)
    edge_features = jnp.array(ef)

    n_node = jnp.array([len(nx_graph_c.nodes())])
    n_edge = jnp.array([len(attacks)])

    global_context = None

    graph = jraph.GraphsTuple(
        nodes=node_features,
        edges=edge_features,
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        globals=global_context
    )
    return graph


def graph_to_jraph(nx_graph):
    mapping = {}
    i = 0
    for n in nx_graph.nodes():
        mapping[n] = i
        i += 1

    # arbitrary int ids to ordered 0,1,2,... ids
    nx_graph_c = nx.relabel_nodes(nx_graph, mapping, copy=True)
    nl_nodes = [nx_graph_c.nodes[n]['text_cat'] for n in nx_graph_c.nodes()]

    # generate sentence embeddings from the text_cat feature and use it as node features
    node_features = jnp.array(WE_MODEL.encode(nl_nodes))

    # define senders and receivers based on the (conflicting) edges of the nx graph
    ef = []
    snd = []
    rec = []
    attacks = [(u, v) for u, v, e in nx_graph_c.edges(data=True) if e['color'] == 'r']
    for att in attacks:
        snd.append(att[0])
        rec.append(att[1])
        ef.append([1.])
    supports = [(u, v) for u, v, e in nx_graph_c.edges(data=True) if e['color'] == 'g']
    for supp in supports:
        snd.append(supp[0])
        rec.append(supp[1])
        ef.append([-1.])
    rephrase = [(u, v) for u, v, e in nx_graph_c.edges(data=True) if e['color'] == 'y']
    for rp in rephrase:
        snd.append(rp[0])
        rec.append(rp[1])
        ef.append([0.])
    senders = jnp.array(snd)
    receivers = jnp.array(rec)
    edge_features = jnp.array(ef)

    n_node = jnp.array([len(nx_graph_c.nodes())])
    n_edge = jnp.array([len(attacks)])

    global_context = None

    graph = jraph.GraphsTuple(
        nodes=node_features,
        edges=edge_features,
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        globals=global_context
    )
    return graph


def convert_networkx_dataset_to_jraph(nx_dataset):
    """Converts a networkx dataset to a jraph graph dataset."""
    jraph_dataset = []
    n = 0
    print("Converting", len(nx_dataset), "samples to Jraph.")
    for nx_sample in nx_dataset:
        # print("Sample ", n, nx_sample)
        sample = {}
        sample['input_graph'] = networkx_to_jraph(nx_sample[0])
        sample['target'] = jnp.array([nx_sample[1]])
        jraph_dataset.append(sample)
        n += 1
    return jraph_dataset


def convert_graph_dataset_to_jraph(g_dataset):
    """Converts a networkx dataset to a jraph graph dataset."""
    jraph_dataset = []
    n = 0
    print("Converting", len(g_dataset), "samples to Jraph.")
    for g_sample in g_dataset:
        # print("Sample ", n, nx_sample)
        sample = {}
        sample['input_graph'] = graph_to_jraph(g_sample[0])
        sample['target'] = jnp.array([g_sample[1]])
        jraph_dataset.append(sample)
        n += 1
    return jraph_dataset


def _nearest_bigger_power_of_two(x: int) -> int:
    """Computes the nearest power of two greater than x for padding."""
    y = 2
    while y < x:
        y *= 2
    return y


def pad_graph_to_nearest_power_of_two(
    graphs_tuple: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Pads a batched `GraphsTuple` to the nearest power of two.
    For example, if a `GraphsTuple` has 7 nodes, 5 edges and 3 graphs, this method
    would pad the `GraphsTuple` nodes and edges:
    7 nodes --> 8 nodes (2^3)
    5 edges --> 8 edges (2^3)
    And since padding is accomplished using `jraph.pad_with_graphs`, an extra
    graph and node is added:
    8 nodes --> 9 nodes
    3 graphs --> 4 graphs
    Args:
    graphs_tuple: a batched `GraphsTuple` (can be batch size 1).
    Returns:
    A graphs_tuple batched to the nearest power of two.
    """
    # Add 1 since we need at least one padding node for pad_with_graphs.
    pad_nodes_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_node)) + 1
    pad_edges_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_edge))
    # Add 1 since we need at least one padding graph for pad_with_graphs.
    # We do not pad to nearest power of two because the batch size is fixed.
    pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
    return jraph.pad_with_graphs(graphs_tuple, pad_nodes_to, pad_edges_to,
                               pad_graphs_to)


# Adapted from https://github.com/deepmind/jraph/blob/master/jraph/ogb_examples/train.py
@jraph.concatenated_args
def edge_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
    """Edge update function for graph net."""
    net = hk.Sequential(
      [hk.Linear(128), jax.nn.relu,
       hk.Linear(128)])
    return net(feats)


@jraph.concatenated_args
def node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
    """Node update function for graph net."""
    net = hk.Sequential(
      [hk.Linear(128), jax.nn.relu,
       hk.Linear(128)])
    return net(feats)


@jraph.concatenated_args
def update_global_fn(feats: jnp.ndarray) -> jnp.ndarray:
    """Global update function for graph net."""
    net = hk.Sequential(
      [hk.Linear(128), jax.nn.relu,
       hk.Linear(2)])
    return net(feats)


def net_fn(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    # Add a global paramater for graph classification.
    graph = graph._replace(globals=jnp.zeros([graph.n_node.shape[0], 1]))
    embedder = jraph.GraphMapFeatures(
      hk.Linear(128), hk.Linear(128), hk.Linear(128))
    net = jraph.GraphNetwork(
      update_node_fn=node_update_fn,
      update_edge_fn=edge_update_fn,
      update_global_fn=update_global_fn)
    return net(embedder(graph))


def predict(params: jnp.ndarray, graph: jraph.GraphsTuple, label: jnp.ndarray, net: jraph.GraphsTuple) -> jnp.ndarray:
    """Computes graph prediction."""
    pred_graph = net.apply(params, graph)
    preds = jax.nn.softmax(pred_graph.globals)
    targets = jax.nn.one_hot(label, 2)

    return preds


def compute_loss(params: jnp.ndarray, graph: jraph.GraphsTuple, label: jnp.ndarray, net: jraph.GraphsTuple) -> jnp.ndarray:
    """Computes loss and accuracy."""
    pred_graph = net.apply(params, graph)
    preds = jax.nn.log_softmax(pred_graph.globals)
    targets = jax.nn.one_hot(label, 2)

    # Since we have an extra 'dummy' graph in our batch due to padding, we want
    # to mask out any loss associated with the dummy graph.
    # Since we padded with `pad_with_graphs` we can recover the mask by using
    # get_graph_padding_mask.
    mask = jraph.get_graph_padding_mask(pred_graph)

    # Cross entropy loss.
    loss = -jnp.mean(preds * targets * mask[:, None])

    # Accuracy taking into account the mask.
    accuracy = jnp.sum(
      (jnp.argmax(pred_graph.globals, axis=1) == label) * mask)/jnp.sum(mask)
    return loss, accuracy


def train(dataset, num_train_steps: int) -> jnp.ndarray:
    """Training loop."""

    # Transform impure `net_fn` to pure functions with hk.transform.
    net = hk.without_apply_rng(hk.transform(net_fn))
    # Get a candidate graph and label to initialize the network.
    graph = dataset[0]['input_graph']

    # Initialize the network.
    params = net.init(jax.random.PRNGKey(42), graph)
    # Initialize the optimizer.
    opt_init, opt_update = optax.adam(1e-4)
    opt_state = opt_init(params)

    compute_loss_fn = functools.partial(compute_loss, net=net)
    # We jit the computation of our loss, since this is the main computation.
    # Using jax.jit means that we will use a single accelerator. If you want
    # to use more than 1 accelerator, use jax.pmap. More information can be
    # found in the jax documentation.
    compute_loss_fn = jax.jit(jax.value_and_grad(
      compute_loss_fn, has_aux=True))
    print('Training model...')
    for idx in range(num_train_steps):
        graph = dataset[idx % len(dataset)]['input_graph']
        label = dataset[idx % len(dataset)]['target']
        # Jax will re-jit your graphnet every time a new graph shape is encountered.
        # In the limit, this means a new compilation every training step, which
        # will result in *extremely* slow training. To prevent this, pad each
        # batch of graphs to the nearest power of two. Since jax maintains a cache
        # of compiled programs, the compilation cost is amortized.
        graph = pad_graph_to_nearest_power_of_two(graph)

        # Since padding is implemented with pad_with_graphs, an extra graph has
        # been added to the batch, which means there should be an extra label.
        label = jnp.concatenate([label, jnp.array([0])])

        (loss, acc), grad = compute_loss_fn(params, graph, label)
        updates, opt_state = opt_update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        # if idx % 50 == 0:
            # print(f'step: {idx}, loss: {loss}, acc: {acc}')
    print('Training finished')
    return params


def evaluate(dataset, params):
    """Evaluation Script."""
    # Transform impure `net_fn` to pure functions with hk.transform.
    net = hk.without_apply_rng(hk.transform(net_fn))
    # Get a candidate graph and label to initialize the network.
    graph = dataset[0]['input_graph']
    accumulated_loss = 0
    accumulated_accuracy = 0
    compute_loss_fn = jax.jit(functools.partial(compute_loss, net=net))
    print('Evaluating model...')
    for idx in range(len(dataset)):
        graph = dataset[idx]['input_graph']
        label = dataset[idx]['target']
        graph = pad_graph_to_nearest_power_of_two(graph)
        label = jnp.concatenate([label, jnp.array([0])])
        loss, acc = compute_loss_fn(params, graph, label)
        accumulated_accuracy += acc
        accumulated_loss += loss
        # print(f'Evaluated {idx + 1} graphs')
    print('Completed evaluation.')
    loss = accumulated_loss / idx
    accuracy = accumulated_accuracy / idx
    print(f'Eval loss: {loss}, accuracy {accuracy}')
    return loss, accuracy


def preds(dataset, params):
    # Transform impure `net_fn` to pure functions with hk.transform.
    net = hk.without_apply_rng(hk.transform(net_fn))
    pred_list = []
    for idx in range(len(dataset)):
        graph = dataset[idx]['input_graph']
        label = dataset[idx]['target']
        pred = predict(params, graph, label, net)
        pred_list.append([pred, label])

    return pred_list
