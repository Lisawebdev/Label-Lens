
from io import BytesIO

import numpy as np
import pandas as pd

from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, send_file
)

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)
app.secret_key = "replace-this-with-a-random-secret"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# Local prototype: results stay in memory while the server runs.
results_df = None
feature_columns = []
target_name = None


def analyze_dataset(data, target):
    if target not in data.columns:
        raise ValueError("Select a valid target-label column.")

    if len(data) < 25:
        raise ValueError("Upload at least 25 data records.")

    if data[target].isna().any():
        raise ValueError("The target column contains missing values.")

    X = data.drop(columns=[target])
    y_original = data[target]

    if X.shape[1] == 0:
        raise ValueError("The dataset needs at least one feature.")

    if not all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns):
        raise ValueError(
            "All feature columns must be numeric in this version."
        )

    if X.isna().any().any():
        raise ValueError("Feature columns contain missing values.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Features must contain finite numeric values.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_original)

    if len(encoder.classes_) != 2:
        raise ValueError(
            "This version supports binary classification only."
        )

    class_counts = np.bincount(y)
    if class_counts.min() < 5:
        raise ValueError(
            "Each class needs at least 5 records for 5-fold validation."
        )

    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )

    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42)
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    # Out-of-fold predictions: each row is predicted by a model
    # that was not trained on that row.
    lr_pred = cross_val_predict(lr, X, y, cv=cv)
    rf_pred = cross_val_predict(rf, X, y, cv=cv)

    model_score = (
        (lr_pred != y).astype(float)
        + (rf_pred != y).astype(float)
    ) / 2

    # Standardize features before computing nearest neighbors.
    X_scaled = StandardScaler().fit_transform(X)

    k = min(10, len(data) - 1)

    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X_scaled)
    _, indices = nn.kneighbors(X_scaled)

    neighborhood_scores = []

    for i, neighbor_ids in enumerate(indices):
        other_ids = neighbor_ids[neighbor_ids != i][:k]
        score = np.mean(y[other_ids] != y[i])
        neighborhood_scores.append(score)

    neighborhood_score = np.array(neighborhood_scores)

    combined_score = (
        0.5 * model_score + 0.5 * neighborhood_score
    )

    # Store original labels for display, not encoded labels.
    result = pd.DataFrame({
        "record_id": np.arange(len(data)),
        "supplied_label": y_original.to_numpy(),
        "lr_prediction": encoder.inverse_transform(lr_pred),
        "rf_prediction": encoder.inverse_transform(rf_pred),
        "disagreement_score": model_score,
        "neighborhood_score": neighborhood_score,
        "combined_score": combined_score,
        "review_status": "Needs review",
        "review_notes": ""
    })

    # Keep original feature values available in the exported report.
    for column in X.columns:
        result[column] = X[column].to_numpy()

    return result


@app.route("/", methods=["GET", "POST"])
def index():
    global results_df, feature_columns, target_name

    if request.method == "POST":
        uploaded_file = request.files.get("file")
        target = request.form.get("target", "").strip()

        if not uploaded_file or not uploaded_file.filename:
            flash("Please select a CSV file.")
            return redirect(url_for("index"))

        if not uploaded_file.filename.lower().endswith(".csv"):
            flash("Please upload a CSV file.")
            return redirect(url_for("index"))

        try:
            data = pd.read_csv(uploaded_file)

            if data.empty:
                raise ValueError("The uploaded CSV is empty.")

            results_df = analyze_dataset(data, target)
            feature_columns = [
                c for c in data.columns if c != target
            ]
            target_name = target

            flash(
                f"Analysis complete: {len(results_df)} records analyzed."
            )

        except Exception as exc:
            flash(f"Could not analyze the dataset: {exc}")

        return redirect(url_for("index"))

    preview = []
    total_records = 0
    flagged_count = 0

    if results_df is not None:
        total_records = len(results_df)
        flagged_count = int(
            (results_df["combined_score"] >= 0.5).sum()
        )

        preview = (
            results_df
            .sort_values("combined_score", ascending=False)
            .head(100)
            .to_dict(orient="records")
        )

    return render_template(
        "index.html",
        preview=preview,
        total_records=total_records,
        flagged_count=flagged_count,
        feature_columns=feature_columns,
        target_name=target_name
    )


@app.route("/review", methods=["POST"])
def review():
    global results_df

    if results_df is None:
        flash("Upload a dataset before reviewing records.")
        return redirect(url_for("index"))

    try:
        record_id = int(request.form["record_id"])
        decision = request.form["review_status"]
        note = request.form.get("review_notes", "").strip()

        valid_decisions = {
            "Confirmed incorrect",
            "Valid label",
            "Needs review"
        }

        if decision not in valid_decisions:
            raise ValueError("Invalid review decision.")

        matching = results_df.index[
            results_df["record_id"] == record_id
        ]

        if len(matching) == 0:
            raise ValueError("Record not found.")

        idx = matching[0]
        results_df.at[idx, "review_status"] = decision
        results_df.at[idx, "review_notes"] = note

        flash(f"Review saved for record {record_id}.")

    except (ValueError, KeyError):
        flash("Could not save the review. Check the record and decision.")

    return redirect(url_for("index"))


@app.route("/export")
def export():
    if results_df is None:
        flash("Analyze a dataset before exporting.")
        return redirect(url_for("index"))

    output = BytesIO()
    output.write(results_df.to_csv(index=False).encode("utf-8"))
    output.seek(0)

    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name="labellens_review_results.csv"
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
