# Sentiment Analysis Neural Network

> A lightweight, browser-based Neural Network application implementing sentiment analysis from scratch using vanilla JavaScript and logistic regression.

**Status:** Educational | **Version:** 1.0.0 | **License:** MIT

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [User Interface](#user-interface)
- [Neural Network Architecture](#neural-network-architecture)
- [Training Process](#training-process)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)

---

## 🎯 Project Overview

This project is a comprehensive, **educational Neural Network web application** that implements binary sentiment analysis (positive/negative classification) entirely in the browser. It covers the complete ML pipeline:

- **Data Processing** → Vocabulary building and text encoding
- **Model Training** → Single-layer perceptron with gradient descent
- **Inference** → Real-time sentiment prediction

**Key Principle:** No external ML libraries are used. All mathematical operations are implemented manually in vanilla JavaScript to facilitate learning.

---

## ✨ Features

- 🧠 **Single-Layer Neural Network**: Implements a perceptron using logistic regression
- 📊 **Bag-of-Words Encoding**: Text vectorization for model input
- 🔧 **Configurable Hyperparameters**: Adjustable epochs and learning rate
- 💾 **Custom Dataset Support**: Upload JSON training datasets
- 🎨 **Modern UI**: Responsive Tailwind CSS interface
- ⚡ **Browser-Based**: No server required; runs entirely on the client
- 🔄 **Real-Time Predictions**: Immediate sentiment analysis and probability scoring

---

## 📁 Project Structure

```
sentiment-analysis-neural-network/
├── index.html          # User interface with Tailwind CSS
├── main.js             # Neural network implementation & logic
├── README.md           # Documentation
└── data.json           # (Optional) Custom training dataset
```

---

## 🚀 Getting Started

### Prerequisites

- Modern web browser (Chrome, Firefox, Edge, Safari)
- No dependencies or installations required

### Running the Application

1. **Open the application:**

   ```bash
   Open index.html in a web browser
   ```

2. **Load Training Data:**
   - Click **Extract** without a file to use the default dataset (20 sentiment words)
   - Or upload a custom JSON file with the format:
     ```json
     [
       { "word": "amazing", "sentiment": 1 },
       { "word": "terrible", "sentiment": 0 }
     ]
     ```

3. **Train the Model:**
   - Set **Epochs** (e.g., 100 or 1000)
   - Set **Learning Rate** (e.g., 0.1)
   - Click **Train Neural Network**

4. **Test Sentiment:**
   - Enter text in the input field
   - Click **Analyze** to see predictions

---

## 🎨 User Interface (`index.html`)

The application provides a clean, single-page interface with three functional sections:

### 1. Load Training Data

- **Upload Option**: Select a custom JSON file containing sentiment-labeled words
- **Fallback Dataset**: If no file is provided, fails to load, or cannot be parsed, the application automatically loads a built-in dataset of 20 common sentiment words
- **Status Feedback**: Real-time success/error messages
- **Format Required**: Array of objects with `word` and `sentiment` properties

**Data Format:**

```json
[
  { "word": "word1", "sentiment": 1 },
  { "word": "word2", "sentiment": 0 }
]
```

Where `sentiment` is `1` (positive) or `0` (negative).

### 2. Train Model

Configure and execute model training:

| Parameter         | Type    | Description                                                     |
| ----------------- | ------- | --------------------------------------------------------------- |
| **Epochs**        | Integer | Number of complete passes through the dataset                   |
| **Learning Rate** | Float   | Adjustment magnitude during weight updates (typically 0.01–1.0) |

The training process iterates through the dataset, adjusting model weights to minimize prediction error. Progress is logged to the browser console.

### 3. Test Sentiment

- **Input**: Arbitrary text strings
- **Output**: Classification (POSITIVE/NEGATIVE) with confidence percentage
- **Visual Feedback**: Green for positive, red for negative

---

## 🧠 Neural Network Architecture (`main.js`)

### Overview

The implementation is a **single-layer neural network** (Perceptron) using binary classification with logistic regression. This model learns a linear decision boundary in the input space.

### 1. Data Preprocessing & Vocabulary

#### Vocabulary Building

When a dataset is loaded, all unique words are extracted into a global `VOCAB` array:

```javascript
VOCAB = DATA.map((item) => item.word);
// Example: ["amazing", "terrible", "great", "bad", ...]
```

#### Text-to-Input Encoding (Bag of Words)

Neural networks require numerical inputs. The `textToInput()` function converts text to a binary vector:

```javascript
function textToInput(text) {
  return VOCAB.map((word) => (text.toLowerCase().includes(word) ? 1 : 0));
}
```

**Example:**

- Text: `"amazing movie"`
- VOCAB: `["amazing", "terrible", "great", "bad"]`
- Encoded: `[1, 0, 0, 0]`

---

### 2. Model Parameters

#### Weights Array

Each vocabulary word is assigned a learnable weight that determines its contribution to the sentiment prediction:

```javascript
let weights = VOCAB.map(() => (Math.random() - 0.5) * 0.1);
// Initialized randomly near zero
```

**Initialization Details:**

- Weights are initialized to random values between -0.05 and +0.05
- Reinitialized whenever new training data is loaded via `extractData()`
- Small random values prevent symmetry during training

**Interpretation:**

- Positive weight → word indicates positive sentiment
- Negative weight → word indicates negative sentiment
- Magnitude → strength of influence

#### Bias Term

```javascript
let bias = 0;
```

The bias shifts the decision boundary and acts as a baseline sentiment score independent of input words.

---

### 3. Core Mechanics

#### Sigmoid Activation Function

Squashes raw scores to a probability between 0 and 1:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

```javascript
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}
```

**Output Interpretation:**

- $\sigma(x) \geq 0.5$ → Positive sentiment
- $\sigma(x) < 0.5$ → Negative sentiment

#### Inference (`predict()`)

Computes the prediction probability for a given input:

1. Start with bias term: `rawScore = bias`
2. Multiply inputs by weights and sum: `rawScore += inputs[i] × weights[i]`
3. Apply sigmoid: `output = σ(rawScore)`

```javascript
function predict(inputs) {
  let rawScore = bias;
  for (let i = 0; i < inputs.length; i++) {
    rawScore += inputs[i] * weights[i];
  }
  return sigmoid(rawScore);
}
```

---

## 📚 Training Process

### Algorithm: Gradient Descent with Delta Rule

#### Step 1: Compute Prediction Error

$$\text{error} = \text{expectedOutput} - \text{prediction}$$

```javascript
const prediction = predict(inputs);
const error = expectedOutput - prediction;
```

#### Step 2: Update Weights

Each weight is adjusted proportionally to its contribution to the error:

$$w_i := w_i + \eta \cdot \text{error} \cdot x_i$$

Where:

- $\eta$ = learning rate (controls update magnitude)
- $\text{error}$ = prediction error
- $x_i$ = input value for the $i$-th word

```javascript
for (let i = 0; i < weights.length; i++) {
  weights[i] += learningRate * error * inputs[i];
}
```

#### Step 3: Update Bias

$$b := b + \eta \cdot \text{error}$$

```javascript
bias += learningRate * error;
```

### Training Loop

The `trainModel()` function iterates through the entire dataset multiple times (epochs):

```javascript
function trainModel(epochs, learningRate) {
  for (let epoch = 0; epoch < epochs; epoch++) {
    const loss = runEpoch(DATA, learningRate);
    console.log(`Epoch ${epoch} | Loss: ${loss.toFixed(6)}`);
  }
}
```

**Loss Function (Mean Squared Error):**

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (\text{error}_i)^2$$

---

## 📖 API Reference

### Global State

| Variable  | Type   | Description                             |
| --------- | ------ | --------------------------------------- |
| `DATA`    | Array  | Training dataset                        |
| `VOCAB`   | Array  | Vocabulary of known words               |
| `weights` | Array  | Model weights (one per vocabulary word) |
| `bias`    | Number | Model bias term                         |

### Functions

#### `extractData()`

Loads training data from an uploaded JSON file or uses fallback dataset.

#### `textToInput(text)`

Converts text string to binary input vector.

- **Parameters:** `text` (string)
- **Returns:** Array of 0s and 1s

#### `predict(inputs)`

Computes sentiment probability.

- **Parameters:** `inputs` (array of numbers)
- **Returns:** Probability between 0 and 1

#### `train(inputs, correctAnswer, learningRate)`

Updates weights based on a single training example using the delta rule.

- **Parameters:**
  - `inputs`: Binary input vector
  - `correctAnswer`: Expected output (0 or 1)
  - `learningRate`: Learning rate scalar
- **Returns:** Squared error (error \* error)

#### `runEpoch(dataset, learningRate)`

Processes one complete epoch through the entire dataset.

- **Parameters:**
  - `dataset`: Array of training examples
  - `learningRate`: Learning rate
- **Returns:** Average loss for the epoch

#### `trainModel(epochs, learningRate)`

Executes multi-epoch training.

- **Parameters:**
  - `epochs`: Number of training iterations
  - `learningRate`: Learning rate

#### `testSentiment(text)`

Performs sentiment analysis and displays results.

- **Parameters:** `text` (string)
- **Updates UI:** Displays prediction and confidence score

---

## 💡 Usage Examples

### Example 1: Basic Training and Testing

```javascript
// 1. Load default data
extractData();

// 2. Train with 500 epochs and learning rate of 0.1
trainModel(500, 0.1);

// 3. Test predictions
testSentiment("amazing movie"); // Output: ~90% positive
testSentiment("terrible experience"); // Output: ~5% positive (negative)
testSentiment("ok I guess"); // Output: ~50% (uncertain)
```

### Example 2: Custom Dataset Upload

Create a `training_data.json` file with the required format:

```json
[
  { "word": "love", "sentiment": 1 },
  { "word": "hate", "sentiment": 0 },
  { "word": "happy", "sentiment": 1 },
  { "word": "sad", "sentiment": 0 },
  { "word": "excellent", "sentiment": 1 },
  { "word": "poor", "sentiment": 0 }
]
```

**Workflow:**

1. Open the application in your browser
2. Click the file input under **Load Training Data**
3. Select your `training_data.json` file
4. Click **Extract** to load the dataset
5. Set **Epochs** and **Learning Rate** values
6. Click **Train Neural Network**
7. Use **Test Sentiment** to evaluate predictions

**Important:** The file must be a JSON array with objects containing `word` (string) and `sentiment` (0 or 1) properties.

---

## 📊 Model Characteristics

| Aspect              | Details                       |
| ------------------- | ----------------------------- |
| **Type**            | Single-layer perceptron       |
| **Input**           | Binary bag-of-words vectors   |
| **Output**          | Probability [0, 1]            |
| **Activation**      | Sigmoid function              |
| **Loss Function**   | Mean squared error (MSE)      |
| **Optimization**    | Gradient descent (delta rule) |
| **Hyperparameters** | Learning rate, epochs         |

### Limitations

- **Linear Decision Boundary**: Cannot learn complex, non-linear patterns
- **Bag-of-Words Limitation**: Ignores word order and context
- **Vocabulary Fixed**: Cannot process words not in training vocab
- **No Hidden Layers**: Limited representational capacity

---

## 🎓 Educational Value

This project demonstrates:

✅ Machine learning pipeline (data loading → training → inference)  
✅ Neural network fundamentals (weights, bias, activation functions)  
✅ Gradient descent optimization  
✅ Binary classification with logistic regression  
✅ Browser-based ML without external libraries

---

## 📝 Notes

- All training data is processed in-memory; no persistent storage
- Browser console shows detailed training logs:
  - Loss values logged every 100 epochs during training
  - Final trained weights printed after training completes
  - Prediction results logged for each `testSentiment()` call
- The sigmoid function maintains numerical stability across the full input range
- Weights are reinitialized whenever new training data is loaded

## Notice
- This is a proof of concept project
- This may have bugs or can effect your device preformence if you use high epochs
- This is done by a first year undergraduate so can be problematic on deferent devices
---

## 📄 License

This project is provided as educational material.
