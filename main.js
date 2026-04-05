let DATA = null;
let VOCAB = [];

async function extractData() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        console.error('No file selected');
        document.getElementById("tdError").classList.remove("hidden");
        document.getElementById("tdError").classList.add("text-red-500");
        document.getElementById("tdError").innerText = "No file selected";
        return;
    }

    try {
        const text = await file.text();
        DATA = JSON.parse(text);
    } catch (error) {
        console.error('Error loading file:', error);
        document.getElementById("tdError").classList.remove("hidden");
        document.getElementById("tdError").classList.add("text-red-500");
        document.getElementById("tdError").innerText = "Error loading file";
        DATA = [
            { word: "amazing", sentiment: 1 },
            { word: "terrible", sentiment: 0 },
            { word: "great", sentiment: 1 },
            { word: "bad", sentiment: 0 },
            { word: "excellent", sentiment: 1 },
            { word: "awful", sentiment: 0 },
            { word: "fantastic", sentiment: 1 },
            { word: "horrible", sentiment: 0 },
            { word: "brilliant", sentiment: 1 },
            { word: "worst", sentiment: 0 },
            { word: "wonderful", sentiment: 1 },
            { word: "boring", sentiment: 0 },
            { word: "enjoyed", sentiment: 1 },
            { word: "hated", sentiment: 0 },
            { word: "perfect", sentiment: 1 },
            { word: "dull", sentiment: 0 },
            { word: "best", sentiment: 1 },
            { word: "waste", sentiment: 0 },
            { word: "loved", sentiment: 1 },
            { word: "disappointing", sentiment: 0 },
        ];
    }

    // Build VOCAB after data is loaded
    VOCAB = DATA.map(item => item.word);
    weights = VOCAB.map(() => (Math.random() - 0.5) * 0.1);
    bias = 0;

    console.log('Data loaded:', DATA);
    document.getElementById("tdError").classList.remove("hidden");
    document.getElementById("tdError").classList.add("text-green-500");
    document.getElementById("tdError").innerText = "Data loaded successfully";
}

function textToInput(text) {
    if (typeof text !== 'string') text = text ? String(text) : "";
    const lower = text.toLowerCase();
    return VOCAB.map(word => lower.includes(word) ? 1 : 0);
}


let weights = VOCAB.map(() => (Math.random() - 0.5) * 0.1);
let bias = 0;

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function predict(inputs) {
    let rawScore = bias;
    for (let i = 0; i < inputs.length; i++) {
        rawScore += inputs[i] * weights[i];
    }
    return sigmoid(rawScore);
}

function train(inputs, correctAnswer, learningRate) {
    const prediction = predict(inputs);
    const error = correctAnswer - prediction;

    for (let i = 0; i < weights.length; i++) {
        weights[i] += learningRate * error * inputs[i];
    }

    bias += learningRate * error;
    return error * error;
}

function runEpoch(dataset, learningRate) {
    let totalLoss = 0;
    for (const example of dataset) {
        const inputs = textToInput(example.word);
        totalLoss += train(inputs, example.sentiment, learningRate);
    }
    return totalLoss / dataset.length;
}

function trainModel(epochs, learningRate) {
    for (let epoch = 0; epoch < epochs; epoch++) {
        const loss = runEpoch(DATA, learningRate);
        if (epoch % 100 === 0) {
            console.log(`Epoch ${epoch} | Loss: ${loss.toFixed(6)}`);
        }
    }
    console.log("Training done!");
    document.getElementById("trainError").classList.remove("hidden");
    document.getElementById("trainError").classList.add("text-green-500");
    document.getElementById("trainError").innerText = "Training done!";
    console.log("Final weights:", VOCAB.map((w, i) => `${w}: ${weights[i].toFixed(3)}`));
}


function testSentiment(text) {
    const inputs = textToInput(text);
    const score = predict(inputs);
    const label = score >= 0.5 ? "POSITIVE" : "NEGATIVE";
    console.log(`"${text}" → ${(score * 100).toFixed(1)}% → ${label}`);
    if (label === "POSITIVE") {
        document.getElementById("result").style.color = "green";
    } else {
        document.getElementById("result").style.color = "red";
    }
    document.getElementById("result").innerText = `"${text}" → ${(score * 100).toFixed(1)}% → ${label}`;
}

// console.log(trainModel(1000, 0.1));

// console.log(testSentiment("amazing movie"));
// console.log(testSentiment("terrible and boring"));
// console.log(testSentiment("loved it"));
// console.log(testSentiment("worst film ever"));