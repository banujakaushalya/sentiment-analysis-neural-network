
## Project Overview

The project is an introductory "from-scratch" machine learning web application. It features a complete pipeline—from dataset processing to model training and active inference—running entirely in the browser using vanilla JavaScript. No external AI libraries (like TensorFlow.js or Brain.js) are used; the math is handled manually.

---

## The User Interface (`index.html`)

The front-end uses HTML structurally enhanced with Tailwind CSS. It provides a single-page application card layout with three main functional pillars:

### Load Training Data

Allows you to upload a custom JSON dataset. If no file is provided or an error occurs during upload, it gracefully defaults to a hardcoded fallback dataset of 20 categorized words (e.g., `"amazing" = 1`, `"terrible" = 0`). It dynamically updates the UI to show success or error states.

### Train Model

Two numeric inputs allow the tweaking of hyperparameters:

- **Epochs** (loops through the dataset)
- **Learning Rate** (the aggressiveness of the algorithm's adjustments)

Upon successful training, it emits a green **"Training done!"** message.

### Test Sentiment

Takes arbitrary user string input, passes it to the mathematical engine, computes the probability, and injects the red (negative) or green (positive) outcome onto the screen.

---

## The Neural Network Architecture (`main.js`)

Under the hood, the application implements a **Single-Layer Neural Network (Perceptron)** using Logistic Regression logic.

### 1. Data Preprocessing & Vocabulary

#### The Vocab

When data is loaded, a global `VOCAB` array is built representing the specific words the model will know.

#### Bag of Words / Encoding (`textToInput`)

Neural networks cannot read strings; they read numbers. The algorithm uses a simple encoding where incoming text is mapped against the entire `VOCAB` array. If a word exists in the vocab, its corresponding index receives a `1`; if not, it gets a `0`.

---

### 2. The Model Parameters

#### Weights (`weights` array)

Initialized randomly near zero. Each word in the vocabulary is assigned a mathematical weight determining its influence.

#### Bias (`bias`)

Initialized at `0`. Represents the base sentiment line before considering any words.

---

### 3. Core Mechanics

#### Activation (Sigmoid Function)

Squeezes any raw number passed to it down to a value strictly between `0` and `1`. This output represents the probability of the sentence being positive.

#### Inference (`predict`)

- Multiplies the encoded inputs by their respective weights  
- Aggregates the total score  
- Adds the bias  
- Passes the result through the sigmoid function  

This produces the final prediction probability.

---

### 4. Training (Gradient Descent)

#### Error Calculation


error = correctAnswer - prediction



#### Weight Updates

Using the delta rule (Gradient Descent), each weight is updated:


weight = weight + (learningRate * error * input)



If a word was heavily positive but the model predicted negative, its weight increases. If incorrect in the opposite direction, the weight decreases.



### Training Loop (`trainModel` & `runEpoch`)

Iterates through the entire dataset repeatedly (controlled by epochs), gradually reducing the loss value until the error stabilizes, resulting in finalized tuned weights.

