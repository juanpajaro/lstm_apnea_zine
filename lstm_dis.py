#!/usr/bin/env python3
"""
Entrenamiento distribuido LSTM con TensorFlow/Keras usando SLURM.

Este script espera que existan los archivos CSV en el directorio actual:
- datos_apnea_train.csv
- datos_apnea_val.csv
- datos_apnea_test.csv

Si se ejecuta con variables de SLURM + WORKER_HOSTS, configura TF_CONFIG
para MultiWorkerMirroredStrategy. Si no, ejecuta en modo local.
"""

import argparse
import csv
import json
import os
import pathlib
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def find_dataset_csv_files(search_root=None):
    """Encuentra los CSV de train/val/test en el directorio actual."""
    root = pathlib.Path(search_root or pathlib.Path.cwd()).resolve()
    dataset_names = (
        "datos_apnea_train.csv",
        "datos_apnea_val.csv",
        "datos_apnea_test.csv",
    )
    dataset_files = {}

    for dataset_name in dataset_names:
        dataset_path = root / dataset_name
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"No se encontro el archivo '{dataset_name}' en el directorio actual: {root}"
            )
        dataset_files[dataset_name] = dataset_path

    return (
        dataset_files["datos_apnea_train.csv"],
        dataset_files["datos_apnea_val.csv"],
        dataset_files["datos_apnea_test.csv"],
    )


def _to_binary_label(value):
    value_str = str(value).strip().lower()
    if value_str in {"1", "true", "yes", "si", "positivo", "positive"}:
        return 1
    if value_str in {"0", "false", "no", "negativo", "negative"}:
        return 0

    try:
        return int(float(value_str))
    except ValueError as exc:
        raise ValueError(f"No se pudo convertir la etiqueta '{value}' a valor numerico") from exc


def _load_csv_as_dataset(
    csv_path,
    batch_size=32,
    shuffle=False,
    text_col="EnfermedadActual",
    label_col="Apnea",
):
    with open(csv_path, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        if len(fieldnames) < 2:
            raise ValueError(
                f"El archivo {csv_path} debe tener al menos 2 columnas (texto y etiqueta)."
            )

        if text_col not in fieldnames or label_col not in fieldnames:
            raise ValueError(
                f"El archivo {csv_path} debe contener las columnas '{text_col}' y '{label_col}'. "
                f"Columnas encontradas: {fieldnames}"
            )

        texts = []
        labels = []
        for row in reader:
            if text_col not in row or label_col not in row:
                raise ValueError(
                    f"Faltan columnas esperadas en {csv_path}. Texto: {text_col}, etiqueta: {label_col}"
                )
            texts.append(str(row[text_col]))
            labels.append(_to_binary_label(row[label_col]))

    if not texts:
        raise ValueError(f"El archivo {csv_path} no contiene filas de datos.")

    dataset = tf.data.Dataset.from_tensor_slices((texts, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(texts), reshuffle_each_iteration=True)

    return dataset.batch(batch_size)


def configure_tf_config_from_slurm():
    """
    Configura TF_CONFIG desde variables de entorno de SLURM.

    Requiere:
    - WORKER_HOSTS=host1:port,host2:port,...
    - SLURM_PROCID

    Si no estan presentes, corre en modo local (worker 0).
    """
    if "TF_CONFIG" in os.environ:
        try:
            tf_config = json.loads(os.environ["TF_CONFIG"])
            task = tf_config.get("task", {})
            return int(task.get("index", 0)), len(tf_config.get("cluster", {}).get("worker", []))
        except Exception:
            return 0, 1

    worker_hosts_env = os.environ.get("WORKER_HOSTS")
    slurm_procid = os.environ.get("SLURM_PROCID")

    if not worker_hosts_env or slurm_procid is None:
        return 0, 1

    worker_hosts = [h.strip() for h in worker_hosts_env.split(",") if h.strip()]
    task_index = int(slurm_procid)

    tf_config = {
        "cluster": {"worker": worker_hosts},
        "task": {"type": "worker", "index": task_index},
    }
    os.environ["TF_CONFIG"] = json.dumps(tf_config)

    return task_index, max(1, len(worker_hosts))


def load_datasets_from_csv(batch_size=32):
    """Carga datasets de train/val/test desde CSV en el directorio actual."""
    train_csv, val_csv, test_csv = find_dataset_csv_files()
    print(f"Train CSV: {train_csv}")
    print(f"Val CSV: {val_csv}")
    print(f"Test CSV: {test_csv}")

    train_ds = _load_csv_as_dataset(
        train_csv,
        batch_size=batch_size,
        shuffle=True,
        text_col="EnfermedadActual",
        label_col="Apnea",
    )
    val_ds = _load_csv_as_dataset(
        val_csv,
        batch_size=batch_size,
        shuffle=False,
        text_col="EnfermedadActual",
        label_col="Apnea",
    )
    test_ds = _load_csv_as_dataset(
        test_csv,
        batch_size=batch_size,
        shuffle=False,
        text_col="EnfermedadActual",
        label_col="Apnea",
    )

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    train_ds = train_ds.with_options(options)
    val_ds = val_ds.with_options(options)
    test_ds = test_ds.with_options(options)

    return train_ds, val_ds, test_ds


def vectorize_datasets(train_ds, val_ds, test_ds, max_tokens=30000, max_length=600):
    """Vectoriza texto a secuencias enteras para el modelo LSTM."""
    text_vectorization = layers.TextVectorization(
        max_tokens=max_tokens,
        split="whitespace",
        output_mode="int",
        output_sequence_length=max_length,
    )

    train_ds_no_labels = train_ds.map(lambda x, y: x)
    text_vectorization.adapt(train_ds_no_labels)

    sequence_train_ds = train_ds.map(
        lambda x, y: (text_vectorization(x), y), num_parallel_calls=tf.data.AUTOTUNE
    )
    sequence_val_ds = val_ds.map(
        lambda x, y: (text_vectorization(x), y), num_parallel_calls=tf.data.AUTOTUNE
    )
    sequence_test_ds = test_ds.map(
        lambda x, y: (text_vectorization(x), y), num_parallel_calls=tf.data.AUTOTUNE
    )

    return (
        sequence_train_ds.prefetch(tf.data.AUTOTUNE),
        sequence_val_ds.prefetch(tf.data.AUTOTUNE),
        sequence_test_ds.prefetch(tf.data.AUTOTUNE),
        max_tokens,
    )


def build_model(max_tokens, hidden_dim=64):
    """Crea y compila el modelo LSTM."""
    model = keras.Sequential(
        [
            layers.Embedding(max_tokens, hidden_dim, name="embedding", mask_zero=True),
            layers.Bidirectional(layers.LSTM(256, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(128)),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_versioned_model(model, output_dir="models"):
    """Guarda el modelo con nombre versionado en formato .keras."""
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_path / f"lstm_model_v{version}.keras"
    model.save(model_path)
    return model_path


def main(per_worker_batch_size=32, epochs=10):
    task_index, num_workers = configure_tf_config_from_slurm()

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    global_batch_size = per_worker_batch_size * max(1, strategy.num_replicas_in_sync)

    print(f"Task index: {task_index}")
    print(f"Workers detectados: {num_workers}")
    print(f"Replicas in sync: {strategy.num_replicas_in_sync}")
    print(f"Global batch size: {global_batch_size}")

    train_ds, val_ds, test_ds = load_datasets_from_csv(batch_size=global_batch_size)
    sequence_train_ds, sequence_val_ds, sequence_test_ds, max_tokens = vectorize_datasets(
        train_ds, val_ds, test_ds
    )

    with strategy.scope():
        model = build_model(max_tokens=max_tokens)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        restore_best_weights=True,
        patience=2,
    )

    model.fit(
        sequence_train_ds,
        validation_data=sequence_val_ds,
        epochs=epochs,
        callbacks=[early_stopping],
    )

    test_loss, test_acc = model.evaluate(sequence_test_ds)

    # Solo el worker principal escribe archivos para evitar conflictos.
    if task_index == 0:
        model_path = save_versioned_model(model)
        with open("performance_report.txt", "w", encoding="utf-8") as report_file:
            report_file.write(f"Test Loss: {test_loss}\n")
            report_file.write(f"Test Accuracy: {test_acc}\n")
            report_file.write(f"Workers: {num_workers}\n")
            report_file.write(f"Global Batch Size: {global_batch_size}\n")

        print(f"\nModelo guardado en: {model_path}")
        print("Reporte guardado en: performance_report.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenamiento distribuido LSTM con TensorFlow/Keras"
    )
    parser.add_argument(
        "--per-worker-batch-size",
        type=int,
        default=32,
        help="Batch size por worker (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Numero de epocas de entrenamiento (default: 10)",
    )

    args = parser.parse_args()
    main(per_worker_batch_size=args.per_worker_batch_size, epochs=args.epochs)
