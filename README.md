
# 🧠 Backpropagation Engine from Scratch

This notebook implements a **mini automatic differentiation engine** (aka autograd) entirely from scratch in pure Python and NumPy.

### ✍️ Author Note

I built this project a few months ago while **following along with Andrej Karpathy's 'Neural Networks: Zero to Hero' series**.  
It helped me deeply understand how backpropagation works under the hood, by actually coding out a small system like PyTorch’s `Tensor` class.

---

### 🚀 Features

- Custom `Value` class tracks operations and computes gradients
- Operator overloading: `+`, `*`, `**`, `tanh`, `exp`, `-`, `/`, etc.
- Full backward propagation logic
- Topological sorting for dependency tracking
- Graph visualization with Graphviz

---

### 📸 Output Sample

Once you run a simple forward pass and `.backward()`, you can render the graph with:
```python
from graphviz import Source
dot = draw_dot(L)
dot.render("grad_graph", format="png", cleanup=False)
```

---

### 🧪 Example

```python
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'

L.backward()
```

---

### 📎 Requirements

- Python 3.6+
- numpy
- graphviz (optional, for graph rendering)

You can install dependencies with:
```bash
pip install numpy graphviz
```

---

### 📚 Credit

Based on [Andrej Karpathy’s micrograd](https://github.com/karpathy/micrograd) and his amazing **Zero to Hero** series.  
This version was hand-coded from scratch as part of my learning journey.

---

### 📂 File

- `Backprop_Cleaned_For_GitHub.ipynb`: Cleaned notebook version, ready to run and explore.

