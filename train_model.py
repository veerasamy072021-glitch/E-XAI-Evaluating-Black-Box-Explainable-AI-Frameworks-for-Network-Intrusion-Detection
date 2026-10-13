from __future__ import annotations

from modules.config import DB_PATH, ensure_directories
from modules.database import initialize_database, save_model_metrics
from modules.dataset_manager import load_best_available_dataset
from modules.model_engine import save_training_result, train_intrusion_model


def main() -> None:
    ensure_directories()
    frame, info = load_best_available_dataset(prefer_real=True)
    result = train_intrusion_model(frame, info)
    save_training_result(result)
    initialize_database(DB_PATH)
    save_model_metrics(info["name"], result.metadata["model_name"], result.metrics)
    mode = "DEMO DATA" if info["mode"] == "demo" else "REAL/USER DATA"
    print(f"Training complete [{mode}]")
    print(f"Model: {result.metadata['model_name']}")
    print(f"Samples: {result.metadata['samples']} | Features: {result.metadata['model_feature_count']}")
    print(" | ".join(f"{name.upper()}: {result.metrics[name]:.4f}" for name in ("accuracy", "precision", "recall", "f1", "roc_auc")))


if __name__ == "__main__":
    main()

