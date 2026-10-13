from __future__ import annotations

from modules.agreement_engine import top_k_agreement
from modules.database import read_experiments, save_experiment
from modules.lime_engine import LimeEngine
from modules.model_engine import load_artifacts
from modules.permutation_engine import calculate_permutation_importance
from modules.prediction_engine import predict_transformed
from modules.shap_engine import ShapEngine
from modules.xai_evaluator import evaluate_explainer


def main() -> None:
    artifacts = load_artifacts()
    model = artifacts["model"]
    train = artifacts["train_reference"]["X"]
    row = artifacts["test_reference"]["X"][0]
    names = artifacts["feature_names"]
    shap_engine = ShapEngine(model, train, names)
    lime_engine = LimeEngine(model, train, names)
    shap_table, _ = shap_engine.explain_local(row, 10)
    lime_table, _ = lime_engine.explain_local(row, 10, 800)
    shap_metrics, _ = evaluate_explainer("SHAP", model, row, shap_engine.explain_vector, train, perturbations=3)
    lime_metrics, _ = evaluate_explainer(
        "LIME", model, row, lambda x: lime_engine.explain_vector(x, 500), train, perturbations=3
    )
    agreement = top_k_agreement(shap_table, lime_table, 8)
    permutation_table, permutation_runtime = calculate_permutation_importance(
        model,
        artifacts["test_reference"]["X"],
        artifacts["test_reference"]["y"],
        names,
        n_repeats=3,
    )
    prediction = predict_transformed(model, row)
    for metrics, table in ((shap_metrics, shap_table), (lime_metrics, lime_table)):
        metrics["top_features"] = table["feature"].head(10).tolist()
    record = {
        "dataset": artifacts["metadata"]["dataset"]["name"],
        "model": artifacts["metadata"]["model_name"],
        "sample_id": "smoke-test-0",
        "prediction": prediction["label"],
        "attack_probability": prediction["attack_probability"],
        "risk": prediction["risk"],
        "agreement_jaccard": agreement["jaccard"],
        "agreement_spearman": agreement["rank_correlation"],
        "details": {"purpose": "end-to-end smoke test"},
    }
    experiment_id = save_experiment(record, [shap_metrics, lime_metrics])
    rows = read_experiments()
    if experiment_id not in rows["id"].values:
        raise RuntimeError("Saved experiment was not found during database read-back.")
    print(f"E-XAI smoke test passed. Experiment {experiment_id} saved and read back.")
    print(f"Prediction: {prediction['label']} | attack probability: {prediction['attack_probability']:.4f}")
    print(f"SHAP quality: {shap_metrics['quality']:.4f} | LIME quality: {lime_metrics['quality']:.4f}")
    print(f"Top-K Jaccard: {agreement['jaccard']:.4f}")
    print(f"Permutation importance: {len(permutation_table)} features in {permutation_runtime:.4f}s")


if __name__ == "__main__":
    main()
