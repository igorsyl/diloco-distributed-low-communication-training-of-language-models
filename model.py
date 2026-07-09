"""
DiLoCo: Distributed Low-Communication Training of Language Models

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_model_params
def init_model_params(input_dim, hidden_dim, output_dim, seed=0):
    # return dict with W1 (in,hid), b1 (hid,), W2 (hid,out), b2 (out,) as float64
    rng = np.random.default_rng(seed)
    return {
        'W1': rng.standard_normal((input_dim, hidden_dim)) * np.sqrt(2/input_dim),
        'W2': rng.standard_normal((hidden_dim, output_dim)) * np.sqrt(2/hidden_dim),
        'b1': np.zeros(hidden_dim),
        'b2': np.zeros(output_dim),
    }

# Step 2 - relu
import numpy as np

def relu(x):
    # return an array of the same shape as x with negatives clipped to 0.
    return np.maximum(x, 0)

# Step 3 - model_forward
import numpy as np

def model_forward(params, x):
    """Run the 2-layer MLP forward pass and stash intermediates for backprop."""
    # compute z1, h1 = relu(z1), logits, and return (logits, cache).
    z1 = x @ params['W1'] + params['b1']
    h1 = relu(z1)
    logits = h1 @ params['W2'] + params['b2']
    cache = {
        'h1': h1,
        'logits': logits,
        'x': x,
        'z1': z1,
    }
    return logits, cache

# Step 4 - softmax
import numpy as np

def softmax(logits):
    # return a row-wise, numerically-stable softmax of logits with shape (N, C).
    m = np.max(logits, axis=-1, keepdims=True)
    e = np.exp(logits - m)
    s = np.sum(e, axis=-1, keepdims=True)
    return e/s

# Step 5 - cross_entropy_loss
def cross_entropy_loss(logits, labels):
    # Compute the mean cross-entropy loss between logits (N,C) and integer labels (N,).
    probs = softmax(logits)
    N = labels.shape[0]
    return -np.mean(np.log(np.clip(probs[np.arange(N), labels], 1e-50, 1.0)))

# Step 6 - model_backward
def model_backward(params, cache, labels):
    # return a dict of gradients {'W1','b1','W2','b2'} from cache and labels.
    x, z1, h1, logits = cache['x'], cache['z1'], cache['h1'], cache['logits']
    W2 = params['W2']
    p = softmax(logits)
    N = labels.shape[0]
    dlogits = p.copy()
    dlogits[np.arange(N), labels] -= 1
    dlogits /= N
    dW2 = h1.T @ dlogits
    db2 = dlogits.sum(axis=0)
    dh1 = dlogits @ W2.T
    dz1 = dh1 * (z1 > 0)
    dW1 = x.T @ dz1
    db1 = dz1.sum(axis=0)
    return {
        'W1': dW1,
        'b1': db1,
        'W2': dW2,
        'b2': db2,
    }

# Step 7 - init_adamw_state
def init_adamw_state(params):
    # Build the AdamW state dict with zeroed first/second moments and t=0.
    m = {}
    v = {}
    for key,val in params.items():
        m[key] = np.zeros_like(val)
        v[key] = np.zeros_like(val)
    t = 0
    state = {
        'm': m, 'v': v, 't': t,
    }
    return state

# Step 8 - update_adam_moments
def update_adam_moments(state, grads, beta1, beta2):
    # increment state['t'] and update first/second moment EMAs for each param key.
    m, v, t = state['m'], state['v'], state['t']
    for key, grad in grads.items():
        m[key] = beta1 * m[key] + (1-beta1)*grad
        v[key] = beta2 * v[key] + (1-beta2)*grad*grad
    state['t'] += 1
    return state

# Step 9 - bias_correct_moments
def bias_correct_moments(state, beta1, beta2):
    # return (m_hat, v_hat) dicts with Adam bias-corrected moments at step state['t'].
    m, v, t = state['m'], state['v'], state['t']
    m_hat = {}
    v_hat = {}
    for k in m.keys():
        m_hat[k] = m[k] / (1-beta1**t)
        v_hat[k] = v[k] / (1-beta2**t)

    return (m_hat, v_hat)

# Step 10 - adam_param_step
def adam_param_step(params, m_hat, v_hat, lr, eps):
    # return new params updated by p - lr * m_hat / (sqrt(v_hat) + eps) elementwise.
    return {p: params[p] - lr * m_hat[p] / (np.sqrt(v_hat[p]) + eps) for p in params.keys()}

# Step 11 - decoupled_weight_decay
import numpy as np

def decoupled_weight_decay(params, lr, weight_decay):
    # return a new params dict where each tensor is shrunk by AdamW's decoupled weight decay factor.
    return {p: params[p] * (1 - lr * weight_decay) for p in params.keys()}

# Step 12 - clone_params
def clone_params(params):
    # return a new dict whose values are independent copies of the input arrays.
    return {k: v.copy() for k, v in params.items()}

# Step 13 - scale_params
def scale_params(params, scalar):
    # return a new dict where every array in params is multiplied by scalar.
    return {k: scalar * v for k,v in params.items()}

# Step 14 - subtract_params
def subtract_params(params_a, params_b):
    # return a new dict with params_a[k] - params_b[k] for each key.
    return {k: params_a[k]-params_b[k] for k in params_a.keys()}

# Step 15 - average_params
def average_params(params_list):
    # return a new dict whose value at each key is the element-wise mean across the input dicts.
    return {k: np.mean([p[k] for p in params_list], axis=0) for k in params_list[0].keys()}

# Step 16 - iid_shard_dataset
def iid_shard_dataset(x, y, num_workers, seed=0):
    # partition (x, y) uniformly at random into num_workers disjoint IID shards.
    N,D = x.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    splits = np.array_split(perm, num_workers)
    out = []
    for i in range(num_workers):
        x_shard = x[splits[i]]
        y_shard = y[splits[i]]
        out.append((x_shard, y_shard))
    return out

# Step 17 - noniid_shard_dataset
def noniid_shard_dataset(x, y, num_workers, num_classes, seed=0):
    # partition (x, y) across workers so each worker owns a distinct subset of classes.
    owned = [[] for w in range(num_workers)]
    for c in range(num_classes):
        owner = c % num_workers
        owned[owner].append(c)

    rng = np.random.default_rng(seed)
    shards = []
    for w in range(num_workers):
        classes = owned[w]
        mask = np.isin(y, classes)
        idx = np.where(mask)[0]
        rng.shuffle(idx)
        shard = x[idx], y[idx]
        shards.append(shard)

    return shards

# Step 18 - sample_worker_batch
def sample_worker_batch(x_shard, y_shard, batch_size, rng):
    # Sample batch_size examples from (x_shard, y_shard) using rng and return (x_batch, y_batch).
    n, D = x_shard.shape
    replace = batch_size > n
    idx = rng.choice(n, size=batch_size, replace=replace)
    return x_shard[idx], y_shard[idx]

# Step 19 - local_train_step
def local_train_step(params, adam_state, x_batch, y_batch, lr, beta1, beta2, eps, weight_decay):
    # one AdamW update: forward, loss, backward, moment update, param step, weight decay.
    logits, cache = model_forward(params, x_batch)
    loss = cross_entropy_loss(logits, y_batch)
    grads = model_backward(params, cache, y_batch)
    adam_state = update_adam_moments(adam_state, grads, beta1, beta2)
    (m_hat, v_hat) = bias_correct_moments(adam_state, beta1, beta2)
    params = adam_param_step(params, m_hat, v_hat, lr, eps)
    params = decoupled_weight_decay(params, lr, weight_decay)
    return (params, adam_state, loss)

# Step 20 - inner_train_worker
def inner_train_worker(params, x_shard, y_shard, num_inner_steps, batch_size, lr, beta1, beta2, eps, weight_decay, seed):
    # run num_inner_steps AdamW updates on this worker's shard from a copy of params
    params = clone_params(params)
    adam_state = init_adamw_state(params)
    rng = np.random.default_rng(seed)
    mean_loss = 0
    for i in range(num_inner_steps):
        (x_batch, y_batch) = sample_worker_batch(x_shard, y_shard, batch_size, rng)
        (params, adam_state, loss) = local_train_step(params, adam_state, x_batch, y_batch, lr, beta1, beta2, eps, weight_decay)
        mean_loss += loss / num_inner_steps
    return params, mean_loss

# Step 21 - init_outer_optimizer
def init_outer_optimizer(params):
    # Build outer optimizer state with a zero momentum buffer matching params shapes.
    m = {}
    for p, v in params.items():
        m[p] = np.zeros_like(v)
    state = {
        'momentum': m
    }
    return state

# Step 22 - update_outer_momentum
import numpy as np

def update_outer_momentum(outer_state, outer_grad, momentum_coef):
    """Update Nesterov momentum buffer: m <- momentum_coef * m + outer_grad."""
    # for each key in outer_state['momentum'], set m[k] = momentum_coef * m[k] + outer_grad[k]
    momentum = outer_state['momentum']
    new_momentum = {}
    for k,v in momentum.items():
        new_momentum[k] = momentum_coef * momentum[k] + outer_grad[k]
    outer_state = {
        'momentum': new_momentum
    }
    return outer_state

# Step 23 - nesterov_param_update
def nesterov_param_update(params, outer_state, outer_grad, outer_lr, momentum_coef):
    # apply the Nesterov look-ahead step using the (already-updated) momentum buffer and outer_grad.
    new_params = {}
    for k in params.keys():
        u_k = momentum_coef * outer_state[k] + outer_grad[k]
        new_params[k] = params[k] - outer_lr * u_k
    return new_params

# Step 24 - compute_outer_gradient
def compute_outer_gradient(global_params, worker_params_list):
    # return global_params minus the elementwise mean of worker_params_list, key by key.
    return subtract_params(global_params, average_params(worker_params_list))

# Step 25 - run_diloco_round
def run_diloco_round(global_params, outer_state, worker_shards, num_inner_steps, batch_size, inner_hparams, outer_lr, momentum_coef, seed):
    # run inner training on each worker, aggregate pseudo-gradient, apply Nesterov outer step.
    lr, beta1, beta2, eps, weight_decay = inner_hparams['lr'], inner_hparams['beta1'], inner_hparams['beta2'], inner_hparams['eps'], inner_hparams['weight_decay']
    worker_params_list = []
    worker_losses = []
    for i,shard in enumerate(worker_shards):
        x, y = shard
        local_params, mean_loss = inner_train_worker(global_params, x, y, num_inner_steps, batch_size, lr, beta1, beta2, eps, weight_decay, seed+i)
        worker_params_list.append(local_params)
        worker_losses.append(mean_loss)
        
    outer_grad = compute_outer_gradient(global_params, worker_params_list)
    outer_state = update_outer_momentum(outer_state, outer_grad, momentum_coef)
    momentum = outer_state['momentum']
    new_global_parms = nesterov_param_update(global_params, momentum, outer_grad, outer_lr, momentum_coef)
    return (new_global_parms, outer_state, worker_losses)

# Step 26 - train_diloco
def train_diloco(init_params, worker_shards, num_rounds, num_inner_steps, batch_size, inner_hparams, outer_lr, momentum_coef, seed=0):
    # run num_rounds DiLoCo communication rounds and return (final_params, history)
    params = clone_params(init_params)
    outer_state = init_outer_optimizer(params)

    round_losses = []
    for r in range(num_rounds):
        params, outer_state, worker_losses = run_diloco_round(params, outer_state, worker_shards, num_inner_steps, batch_size, inner_hparams, outer_lr, momentum_coef, seed + r)
        round_losses.append(float(np.mean(worker_losses)))

    final_params = params
    hist = {
        'round_losses': round_losses,
    }
    return final_params, hist

# Step 27 - train_synchronous_baseline
def train_synchronous_baseline(init_params, worker_shards, num_steps, batch_size, inner_hparams, seed=0):
    # train synchronously by averaging per-worker gradients into one AdamW step per iteration.
    params = clone_params(init_params)
    adam_state = init_adamw_state(params)
    lr, beta1, beta2, eps, weight_decay = inner_hparams['lr'], inner_hparams['beta1'], inner_hparams['beta2'], inner_hparams['eps'], inner_hparams['weight_decay']
    rng = np.random.default_rng(seed)
    step_losses = []
    for i in range(num_steps):
        mean_worker_loss = 0
        grads_list = []
        for x_shard, y_shard in worker_shards:
            x_batch, y_batch = sample_worker_batch(x_shard, y_shard, batch_size, rng)
            logits, cache = model_forward(params, x_batch)
            loss = cross_entropy_loss(logits, y_batch)
            mean_worker_loss += loss / len(worker_shards)
            grads = model_backward(params, cache, y_batch)
            grads_list.append(grads)
        avg_grads = average_params(grads_list)
        adam_state = update_adam_moments(adam_state, avg_grads, beta1, beta2)
        m_hat, v_hat = bias_correct_moments(adam_state, beta1, beta2)
        params = adam_param_step(params, m_hat, v_hat, lr, eps)
        params = decoupled_weight_decay(params, lr, weight_decay)
        step_losses.append(mean_worker_loss)

    return (params, {'step_losses': step_losses})

# Step 28 - evaluate_loss
def evaluate_loss(params, x, y):
    # return the mean cross-entropy loss of the model on the held-out data (x, y).
    logits, cache = model_forward(params, x)
    loss = cross_entropy_loss(logits, y)
    return float(loss)

# Step 29 - classification_accuracy
def classification_accuracy(params, x, y):
    # return top-1 accuracy of the 2-layer MLP on (x, y) as a float in [0, 1].
    logits, cache = model_forward(params, x)
    preds = np.argmax(logits, axis=-1)
    accuracy = float((preds == y).mean())
    return accuracy

# Step 30 - communication_savings
def communication_savings(num_rounds, num_inner_steps, num_workers, param_count):
    # count scalars transmitted under DiLoCo vs a fully synchronous baseline and return the ratio.
    diloco_scalars = 2 * num_workers * param_count * num_rounds
    sync_scalars = 2 * num_workers * param_count * num_rounds * num_inner_steps
    ratio = diloco_scalars / sync_scalars
    savings_factor = 1 / ratio
    return {
        'diloco_scalars': diloco_scalars,
        'sync_scalars': sync_scalars,
        'ratio': ratio,
        'savings_factor': savings_factor,
    }

