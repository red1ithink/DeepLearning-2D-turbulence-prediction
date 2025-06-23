import os, random, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models

# ────────────────── parameter ────────────────── #
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

tf.random.set_seed(SEED); random.seed(SEED); np.random.seed(SEED)

# Split
all_files = [f'vorticity{c}_rot{ang}_trimmed.npy'
             for c in range(NUM_CASES) for ang in ANGLES]
random.shuffle(all_files)
split_idx   = int(len(all_files) * TRAIN_RATIO)
train_files = all_files[:split_idx]
val_files   = all_files[split_idx:]

# Sequence extraction
def extract_sequences(file_path):
    fname = file_path.decode() if isinstance(file_path, bytes) \
            else file_path.numpy().decode()
    v = np.load(os.path.join(DATA_DIR, fname)).astype(np.float32)
    v /= max(np.sqrt(np.mean(v[0]**2)), 1e-6)        # RMS norm
    for t in range(v.shape[0] - (INPUT_SEQ + PRED_STEPS)):
        x = v[t:t+INPUT_SEQ][..., None]
        y = v[t+INPUT_SEQ:t+INPUT_SEQ+PRED_STEPS][..., None]
        yield x.astype(np.float32), y.astype(np.float32)

def tf_extract_sequences(file_path):
    return tf.data.Dataset.from_generator(
        extract_sequences,
        output_signature=(
            tf.TensorSpec((INPUT_SEQ,128,128,1), tf.float32),
            tf.TensorSpec((PRED_STEPS,128,128,1), tf.float32)
        ),
        args=(file_path,)
    )

def make_dataset(files, shuffle):
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.interleave(tf_extract_sequences,
                       cycle_length=tf.data.AUTOTUNE,
                       num_parallel_calls=tf.data.AUTOTUNE,
                       deterministic=False)
    if shuffle:
        ds = ds.shuffle(1000, seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(train_files, True)
val_ds   = make_dataset(val_files, False)

# Train Model
def build_convlstm():
    inp = layers.Input(shape=(INPUT_SEQ,128,128,1))
    x   = layers.ConvLSTM2D(64,5,padding='same',activation='tanh')(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Conv2D(32,3,padding='same',activation='relu')(x)
    out = layers.Conv2D(1,3,padding='same')(x)
    return models.Model(inp,out)

model     = build_convlstm()
optimizer = tf.keras.optimizers.Adam(LR)
mse_loss  = tf.keras.losses.MeanSquaredError()

# train/val step
@tf.function
def train_step(x_init, y_seq):
    current = x_init                     # (B,3,128,128,1)
    with tf.GradientTape() as tape:
        loss = 0.0
        for t in tf.range(PRED_STEPS):   #EACHER FORCING
            pred = model(current, training=True)
            gt   = y_seq[:, t]
            loss += mse_loss(gt, pred)
            current = tf.concat([current[:,1:], tf.expand_dims(gt,1)], axis=1)
        loss /= tf.cast(PRED_STEPS, tf.float32)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

@tf.function
def val_step(x_init, y_seq):
    current = x_init
    loss = 0.0
    for t in tf.range(PRED_STEPS):
        pred = model(current, training=False)
        gt   = y_seq[:, t]
        loss += mse_loss(gt, pred)
        current = tf.concat([current[:,1:], tf.expand_dims(pred,1)], axis=1)
    return loss / tf.cast(PRED_STEPS, tf.float32)

# Train loop
best_val, wait = float('inf'), 0

for epoch in range(1, EPOCHS+1):
    # ---- train ----
    train_metric = tf.keras.metrics.Mean()
    for xb, yb in train_ds:
        train_metric.update_state(train_step(xb, yb))
    train_loss = train_metric.result().numpy()

    # ---- val ----
    val_metric = tf.keras.metrics.Mean()
    for xb, yb in val_ds:
        val_metric.update_state(val_step(xb, yb))
    val_loss = val_metric.result().numpy()

    print(f"📘 Epoch {epoch:4d}/{EPOCHS} | Train={train_loss:.6f} | Val={val_loss:.6f}")

    if val_loss < best_val:
        best_val, wait = val_loss, 0
        model.save('best_nosched_model.keras')
        print("model saved.")
    else:
        wait += 1
        print(f"⏳ Wait: {wait}/{PATIENCE}")
        if wait >= PATIENCE:
            print("EarlyStopping triggered.")
            break
