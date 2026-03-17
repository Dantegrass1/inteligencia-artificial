import torch
import matplotlib.pyplot as plt

# -----------------------------
# 1. GENERAR DATOS
# -----------------------------
# Generamos muchas ecuaciones aleatorias

N = 500

a = torch.rand(N, 1) * 4 + 1      # evitar a=0
b = torch.rand(N, 1) * 10 - 5
c = torch.rand(N, 1) * 10 - 5

# Discriminante
disc = b**2 - 4*a*c

# Nos quedamos solo con raíces reales
mask = disc >= 0

a = a[mask]
b = b[mask]
c = c[mask]

# 🔥 ARREGLAR FORMA
a = a.view(-1, 1)
b = b.view(-1, 1)
c = c.view(-1, 1)

# Fórmula real (ground truth)
x1 = (-b + torch.sqrt(disc[mask])) / (2*a)
x2 = (-b - torch.sqrt(disc[mask])) / (2*a)

# Entrada y salida
X = torch.cat([a, b, c], dim=1)   # (a,b,c)
Y = torch.cat([x1, x2], dim=1)    # (x1,x2)

# -----------------------------
# 2. MODELO (un poco más potente)
# -----------------------------
model = torch.nn.Sequential(
    torch.nn.Linear(3, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 2)
)

# -----------------------------
# 3. ENTRENAMIENTO
# -----------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

losses = []

for epoch in range(500):

    y_pred = model(X)

    loss = ((y_pred - Y)**2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss {loss.item():.4f}")

# -----------------------------
# 4. VISUALIZAR ERROR
# -----------------------------
plt.plot(losses)
plt.title("Error durante entrenamiento")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# -----------------------------
# 5. PROBAR
# -----------------------------
# Ejemplo: x^2 - 4 = 0 → raíces ±2

test = torch.tensor([[1.0, 0.0, -4.0]])

pred = model(test)

print("Raíces reales: -2 y 2")
print("Predicción del modelo:", pred)