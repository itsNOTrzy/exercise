import math

import numpy as np


np.random.seed(42)


def target_function(x):
    return np.sin(2.0 * x) + 0.3 * x ** 2


def build_dataset(num_samples):
    x = np.linspace(-3.0, 3.0, num_samples, dtype=np.float64).reshape(-1, 1)
    y = target_function(x)
    return x, y


def relu(x):
    return np.maximum(x, 0.0)


def relu_grad(x):
    return (x > 0).astype(np.float64)


class ReLUNetwork:
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.4
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.2
        self.b2 = np.zeros((1, hidden_dim))
        self.W3 = np.random.randn(hidden_dim, output_dim) * 0.2
        self.b3 = np.zeros((1, output_dim))

    def forward(self, x):
        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.h1 = relu(self.z1)
        self.z2 = self.h1 @ self.W2 + self.b2
        self.h2 = relu(self.z2)
        self.y_pred = self.h2 @ self.W3 + self.b3
        return self.y_pred

    def backward(self, y_true):
        batch_size = y_true.shape[0]
        grad_y = (2.0 / batch_size) * (self.y_pred - y_true)

        dW3 = self.h2.T @ grad_y
        db3 = np.sum(grad_y, axis=0, keepdims=True)

        grad_h2 = grad_y @ self.W3.T
        grad_z2 = grad_h2 * relu_grad(self.z2)
        dW2 = self.h1.T @ grad_z2
        db2 = np.sum(grad_z2, axis=0, keepdims=True)

        grad_h1 = grad_z2 @ self.W2.T
        grad_z1 = grad_h1 * relu_grad(self.z1)
        dW1 = self.x.T @ grad_z1
        db1 = np.sum(grad_z1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2, dW3, db3

    def step(self, grads, lr):
        dW1, db1, dW2, db2, dW3, db3 = grads
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W3 -= lr * dW3
        self.b3 -= lr * db3


def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def mae(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))


def make_svg_line_plot(x_true, y_true, y_pred, losses, output_path):
    width, height = 1000, 420
    pad = 40
    panel_w = (width - pad * 3) / 2
    panel_h = height - 2 * pad

    def scale(xs, ys, x_min, x_max, y_min, y_max, x_offset):
        pts = []
        for xv, yv in zip(xs, ys):
            px = x_offset + (xv - x_min) / (x_max - x_min) * panel_w
            py = pad + panel_h - (yv - y_min) / (y_max - y_min) * panel_h
            pts.append(f"{px:.2f},{py:.2f}")
        return " ".join(pts)

    x_vals = x_true[:, 0]
    y_true_vals = y_true[:, 0]
    y_pred_vals = y_pred[:, 0]
    loss_x = np.arange(len(losses))
    loss_y = np.array(losses)

    y_min = min(np.min(y_true_vals), np.min(y_pred_vals))
    y_max = max(np.max(y_true_vals), np.max(y_pred_vals))
    loss_min = float(np.min(loss_y))
    loss_max = float(np.max(loss_y))

    left_true = scale(x_vals, y_true_vals, -3.0, 3.0, y_min, y_max, pad)
    left_pred = scale(x_vals, y_pred_vals, -3.0, 3.0, y_min, y_max, pad)
    right_loss = scale(loss_x, loss_y, 0.0, max(len(losses) - 1, 1), loss_min, loss_max, pad * 2 + panel_w)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{pad}" y="24" font-size="18">Function Fitting</text>
<text x="{pad}" y="40" font-size="12">Target vs Prediction</text>
<text x="{pad * 2 + panel_w}" y="24" font-size="18">Training Loss</text>
<line x1="{pad}" y1="{height - pad}" x2="{pad + panel_w}" y2="{height - pad}" stroke="#999"/>
<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#999"/>
<line x1="{pad * 2 + panel_w}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#999"/>
<line x1="{pad * 2 + panel_w}" y1="{pad}" x2="{pad * 2 + panel_w}" y2="{height - pad}" stroke="#999"/>
<polyline points="{left_true}" fill="none" stroke="#1f77b4" stroke-width="2"/>
<polyline points="{left_pred}" fill="none" stroke="#d62728" stroke-width="2"/>
<polyline points="{right_loss}" fill="none" stroke="#2ca02c" stroke-width="2"/>
<text x="{pad + 10}" y="{pad + 20}" font-size="12" fill="#1f77b4">target</text>
<text x="{pad + 10}" y="{pad + 38}" font-size="12" fill="#d62728">prediction</text>
<text x="{pad * 2 + panel_w + 10}" y="{pad + 20}" font-size="12" fill="#2ca02c">train mse</text>
</svg>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)


def main():
    x_train, y_train = build_dataset(512)
    x_test, y_test = build_dataset(256)

    model = ReLUNetwork()
    learning_rate = 0.003
    epochs = 4000
    batch_size = 64
    losses = []

    for epoch in range(epochs):
        perm = np.random.permutation(len(x_train))
        x_train_shuffled = x_train[perm]
        y_train_shuffled = y_train[perm]

        for start in range(0, len(x_train), batch_size):
            end = start + batch_size
            xb = x_train_shuffled[start:end]
            yb = y_train_shuffled[start:end]
            pred = model.forward(xb)
            grads = model.backward(yb)
            model.step(grads, learning_rate)

        if epoch % 20 == 0 or epoch == epochs - 1:
            train_pred = model.forward(x_train)
            losses.append(float(mse(train_pred, y_train)))

    test_pred = model.forward(x_test)
    test_mse = mse(test_pred, y_test)
    test_mae = mae(test_pred, y_test)

    print(f"test_mse={test_mse:.6f}")
    print(f"test_mae={test_mae:.6f}")

    np.savetxt(
        "function_fitting_predictions.csv",
        np.concatenate([x_test, y_test, test_pred], axis=1),
        delimiter=",",
        header="x,target,prediction",
        comments="",
    )
    make_svg_line_plot(x_test, y_test, test_pred, losses, "function_fitting_result.svg")
    print("prediction_file=function_fitting_predictions.csv")
    print("figure_file=function_fitting_result.svg")


if __name__ == "__main__":
    main()
