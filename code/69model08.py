"""
ConvLSTM  — scheduled-sampling (ε @ epoch10 = 0.80)
"""

import os, random, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models

# Parameter
DATA_DIR   = '../v01/vorticity_rotated_trimmed'
ANGLES     = [0, 90, 180, 270]
NUM_CASES  = 50
INPUT_SEQ  = 3
PRED_STEPS = 5
TRAIN_RATIO= 0.8
BATCH_SIZE = 2
EPOCHS     = 1000
PATIENCE   = 50
LR         = 1e-4
SEED       = 42

tf.random.set_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ε-scheduler
TAU = 120
def get_eps(e):
    if e <= 10:   return np.float32(1.0 - 0.02*e)
    return np.float32(0.8*np.exp(-(e-10)/TAU))

# Split
all_files = [f'vorticity{c}_rot{ang}_trimmed.npy'
             for c in range(NUM_CASES) for ang in ANGLES]
random.shuffle(all_files)
split = int(len(all_files)*TRAIN_RATIO)
train_files, val_files = all_files[:split], all_files[split:]
print(f"▶ Train files: {len(train_files)} | Val files: {len(val_files)}")

# Sequence extraction
def extract_sequences(fp):
    name = fp.decode() if isinstance(fp, bytes) else fp.numpy().decode()
    v = np.load(os.path.join(DATA_DIR, name)).astype(np.float32)
    v /= max(np.sqrt(np.mean(v[0]**2)), 1e-6)
    for t in range(v.shape[0]-(INPUT_SEQ+PRED_STEPS)):
        x = v[t:t+INPUT_SEQ][...,None]
        y = v[t+INPUT_SEQ:t+INPUT_SEQ+PRED_STEPS][...,None]
        yield x, y

def tf_extract_sequences(fp):
    return tf.data.Dataset.from_generator(
        extract_sequences,
        output_signature=(
            tf.TensorSpec((INPUT_SEQ,128,128,1), tf.float32),
            tf.TensorSpec((PRED_STEPS,128,128,1), tf.float32)
        ),
        args=(fp,)
    )

def make_dataset(files, shuffle):
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.interleave(tf_extract_sequences,
                       cycle_length=tf.data.AUTOTUNE,
                       num_parallel_calls=tf.data.AUTOTUNE,
                       deterministic=False)
    if shuffle: ds = ds.shuffle(1000, seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(train_files, True)
val_ds   = make_dataset(val_files,   False)

# Train Model
def build_convlstm():
    i = layers.Input((INPUT_SEQ,128,128,1))
    x = layers.ConvLSTM2D(64,5,padding='same',activation='tanh')(i)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32,3,padding='same',activation='relu')(x)
    o = layers.Conv2D(1,3,padding='same')(x)
    return models.Model(i,o)

model = build_convlstm() if START_EPOCH==1 else \
        tf.keras.models.load_model('best_sched_model5.keras', compile=False)
optimizer = tf.keras.optimizers.Adam(LR)
mse_loss  = tf.keras.losses.MeanSquaredError()

# train / val step
@tf.function
def train_step(x_init, y_seq, eps):
    eps = tf.cast(eps, tf.float32)
    cur = x_init; loss = 0.
    with tf.GradientTape() as tape:
        for t in tf.range(PRED_STEPS):
            pred = model(cur, training=True)
            gt   = y_seq[:, t]
            loss += mse_loss(gt, pred)
            mask = tf.random.uniform([tf.shape(x_init)[0],1,1,1],
                                     seed=SEED, dtype=tf.float32) < eps
            nxt  = tf.where(mask, gt, pred)
            cur  = tf.concat([cur[:,1:], nxt[:,None]], axis=1)
        loss /= tf.cast(PRED_STEPS, tf.float32)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

@tf.function
def val_step(x_init, y_seq):
    cur = x_init; loss = 0.
    for t in tf.range(PRED_STEPS):
        pred = model(cur, training=False)
        gt   = y_seq[:, t]
        loss += mse_loss(gt, pred)
        cur  = tf.concat([cur[:,1:], pred[:,None]], axis=1)
    return loss / tf.cast(PRED_STEPS, tf.float32)

# Training Loop
best_val, wait = float('inf'), 0

for epoch in range(START_EPOCH, EPOCHS+1):
    eps = get_eps(epoch)
    # --- train ---
    train_m = tf.keras.metrics.Mean()
    for xb, yb in train_ds:
        train_m.update_state(train_step(xb, yb, eps))
    train_loss = train_m.result().numpy()

    # --- val ---
    val_m = tf.keras.metrics.Mean()
    for xb, yb in val_ds:
        val_m.update_state(val_step(xb, yb))
    val_loss = val_m.result().numpy()

    print(f"📘 Epoch {epoch:4d}/{EPOCHS} | ε={eps:.3f} | "
          f"Train={train_loss:.6f} | Val={val_loss:.6f}")

    if val_loss < best_val:
        best_val, wait = val_loss, 0
        model.save('best_sched_model5.keras')
        print("Best model saved.")
    else:
        wait += 1
        print(f"⏳ Wait: {wait}/{PATIENCE}")
        if wait >= PATIENCE:
            print("EarlyStopping triggered."); break
