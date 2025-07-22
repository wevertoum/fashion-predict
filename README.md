# 🧠 Fashion MNIST Classifier

This is a full-stack image classification app that predicts clothing categories using a trained Fashion MNIST model. The backend is powered by **Flask** and **TensorFlow**, while the frontend is built with **React**, **Vite**, and **MUI**.

---

## 🚀 Getting Started

### 1. Backend Setup (Flask + TensorFlow)

#### ✅ Prerequisites

- Python 3.9 or later
- [pyenv](https://github.com/pyenv/pyenv) recommended for managing Python versions
- pip

#### 📦 Install Python dependencies

```bash
cd backend
pyenv install 3.9.19 --skip-existing
pyenv local 3.9.19
pip install --upgrade pip
pip install Flask Pillow tensorflow flask-cors
```

````

> ⚠️ Make sure `models/fashion_mnist_model.h5` exists in the correct folder. This is the pre-trained model.

#### ▶️ Run the backend server

```bash
python app.py
```

- The server will start on: `http://127.0.0.1:5000`
- Endpoint: `POST /predict`

  - Accepts: image file
  - Returns: predicted class and confidence

---

### 2. Frontend Setup (React + Vite)

#### ✅ Prerequisites

- Node.js (>= 18)
- [pnpm](https://pnpm.io/) or npm/yarn

#### 📦 Install frontend dependencies

```bash
cd frontend
pnpm install
# or
npm install
```

#### ▶️ Start the frontend development server

```bash
pnpm dev
# or
npm run dev
```

- App will be available at: `http://localhost:5173`

---

## 🖼️ How it works

1. Upload an image (e.g. shoe, pullover).
2. Backend processes and resizes the image to 28x28 grayscale.
3. Model predicts the class using a CNN trained on Fashion MNIST.
4. Frontend displays the prediction and confidence level.

---

## 🛠️ Troubleshooting

- **Flask not found?** Make sure your Python virtual environment is activated.
- **CORS errors?** Confirm both frontend and backend are running on localhost.
- **Model not loading?** Check the `.h5` model path and file integrity.

---

## 📌 Tech Stack

- **Frontend**: React, Vite, MUI, TypeScript
- **Backend**: Flask, TensorFlow, Pillow
- **Model**: Pre-trained CNN on Fashion MNIST
````
