import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


FEATURE_NAMES = [
    'time_pressure_score',
    'health_consciousness_index',
    'waste_propensity_score',
    'ai_receptiveness_score',
]


def visualize_decision_tree(model, feature_names, tree_index=0, out_file=None):
    """Plot a single decision tree from the RandomForest model."""
    estimator = model.estimators_[tree_index]
    plt.figure(figsize=(20, 10))
    plot_tree(estimator,
              feature_names=feature_names,
              class_names=model.classes_,
              filled=True,
              rounded=True,
              impurity=False,
              fontsize=8)
    if out_file:
        plt.savefig(out_file, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_feature_importances(model, feature_names, out_file=None):
    """Plot feature importances of the RandomForest model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(6, 4))
    plt.title('Feature Importances')
    plt.bar(range(len(feature_names)), importances[indices], align='center')
    plt.xticks(range(len(feature_names)), np.array(feature_names)[indices],
               rotation=45, ha='right')
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    model = joblib.load('persona_classifier.joblib')

    os.makedirs('visualizations', exist_ok=True)
    visualize_decision_tree(
        model,
        FEATURE_NAMES,
        out_file=os.path.join('visualizations', 'sample_tree.png'))
    plot_feature_importances(
        model,
        FEATURE_NAMES,
        out_file=os.path.join('visualizations', 'feature_importances.png'))
    print("Visualization images saved in the 'visualizations' directory.")