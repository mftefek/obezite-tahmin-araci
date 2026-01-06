// ONNX model session
let session = null;

// Feature statistics - MATLAB normalize() values
const featureStats = {
    'Gender': {'mean': 1.505921, 'std': 0.500083},
    'Age': {'mean': 24.312600, 'std': 6.345968},
    'Height': {'mean': 1.701677, 'std': 0.093305},
    'Weight': {'mean': 86.586058, 'std': 26.191172},
    'family_history_with_overweight': {'mean': 1.817622, 'std': 0.386247},
    'FAVC': {'mean': 1.883941, 'std': 0.320371},
    'FCVC': {'mean': 2.419043, 'std': 0.533927},
    'NCP': {'mean': 2.685628, 'std': 0.778039},
    'CAEC': {'mean': 2.859308, 'std': 0.468543},
    'SMOKE': {'mean': 1.020843, 'std': 0.142893},
    'CH2O': {'mean': 2.008011, 'std': 0.612953},
    'SCC': {'mean': 1.045476, 'std': 0.208395},
    'FAF': {'mean': 1.010298, 'std': 0.850592},
    'TUE': {'mean': 0.657866, 'std': 0.608927},
    'CALC': {'mean': 3.268593, 'std': 0.515498},
    'MTRANS': {'mean': 3.365230, 'std': 1.261423}
};

// CORRECTED: Categorical mapping in alphabetical order
const categoricalMapping = {
    'Gender': {'Female': 1, 'Male': 2},
    'family_history_with_overweight': {'no': 1, 'yes': 2},
    'FAVC': {'no': 1, 'yes': 2},
    'CAEC': {'Always': 1, 'Frequently': 2, 'no': 3, 'Sometimes': 4},
    'SMOKE': {'no': 1, 'yes': 2},
    'SCC': {'no': 1, 'yes': 2},
    'CALC': {'Always': 1, 'Frequently': 2, 'no': 3, 'Sometimes': 4},
    'MTRANS': {'Automobile': 1, 'Bike': 2, 'Motorbike': 3, 'Public_Transportation': 4, 'Walking': 5}
};

// Obesity classes
const obesityClasses = [
    'Insufficient_Weight',
    'Normal_Weight',
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III'
];

// Load model
async function loadModel() {
    try {
        console.log('Loading model...');
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
        session = await ort.InferenceSession.create('./obzeite.onnx', {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
        
        console.log('✅ Model loaded! Input:', session.inputNames[0], 'Output:', session.outputNames[0]);
        document.getElementById('modelStatus').style.display = 'none';
        document.getElementById('modelLoaded').style.display = 'block';
        
        return true;
    } catch (error) {
        console.error('❌ Model loading error:', error);
        document.getElementById('modelStatus').innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle"></i> Model failed to load: ${error.message}
            </div>
        `;
        return false;
    }
}

// Prepare data compatible with MATLAB reshape
function preprocessInput(formData) {
    // Features in MATLAB order
    const features = [
        'Gender', 'Age', 'Height', 'Weight', 
        'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP',
        'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
        'CALC', 'MTRANS'
    ];
    
    const processed = [];
    const rawValues = [];
    
    for (const feature of features) {
        let value = formData[feature];
        rawValues.push({feature, value});
        
        // Convert height to meters
        if (feature === 'Height') {
            value = value / 100;
        }
        
        let numericValue;
        
        // Convert categorical variables to numeric
        if (categoricalMapping[feature]) {
            const mapping = categoricalMapping[feature];
            if (mapping[value] === undefined) {
                console.warn(`⚠️ Unknown value: ${feature}=${value}`);
                // Use first alphabetical value
                const firstKey = Object.keys(mapping)[0];
                numericValue = mapping[firstKey];
            } else {
                numericValue = mapping[value];
            }
        } else {
            numericValue = parseFloat(value);
        }
        
        // Same operation as MATLAB normalize()
        const mean = featureStats[feature].mean;
        const std = featureStats[feature].std;
        const normalized = (numericValue - mean) / std;
        
        processed.push(normalized);
    }
    
    console.log('📋 Raw values:', rawValues);
    console.log('🔢 Normalized values:', processed.map(v => v.toFixed(4)));
    
    // Reshape as in MATLAB: XTrain = reshape(XTrain', [size(X_numeric,2), 1, 1, size(XTrain,1)]);
    // In JavaScript: [batch, channels, height, width] = [1, 1, 16, 1]
    const floatArray = new Float32Array(processed);
    const tensor = new ort.Tensor('float32', floatArray, [1, 1, 16, 1]);
    
    console.log('🎯 Tensor shape:', tensor.dims);
    return tensor;
}

// Simulate a MATLAB example for testing
async function testMATLABExample() {
    console.log('🧪 Testing MATLAB example...');
    
    // Use normalized values from MATLAB
    // Example: Mean values (all features should be 0)
    const testData = {
        'Gender': 'Female', // 1
        'Age': 24.3126, // normalize: (24.3126 - 24.3126) / 6.345968 = 0
        'Height': 1.701677, // normalize: 0
        'Weight': 86.586058, // normalize: 0
        'family_history_with_overweight': 'yes', // 2
        'FAVC': 'yes', // 2
        'FCVC': 2.419043, // 0
        'NCP': 2.685628, // 0
        'CAEC': 'no', // 3 (alphabetical order)
        'SMOKE': 'no', // 1
        'CH2O': 2.008011, // 0
        'SCC': 'no', // 1
        'FAF': 1.010298, // 0
        'TUE': 0.657866, // 0
        'CALC': 'no', // 3 (alphabetical order)
        'MTRANS': 'Public_Transportation' // 4
    };
    
    // Normalized values of categorical features:
    // Gender: Female=1 → (1-1.505921)/0.500083 = -1.011
    // family_history: yes=2 → (2-1.817622)/0.386247 = 0.472
    // FAVC: yes=2 → (2-1.883941)/0.320371 = 0.362
    // CAEC: no=3 → (3-2.859308)/0.468543 = 0.300
    // SMOKE: no=1 → (1-1.020843)/0.142893 = -0.146
    // SCC: no=1 → (1-1.045476)/0.208395 = -0.218
    // CALC: no=3 → (3-3.268593)/0.515498 = -0.521
    // MTRANS: Public_Transportation=4 → (4-3.365230)/1.261423 = 0.503
    
    try {
        const tensor = preprocessInput(testData);
        const inputName = session.inputNames[0];
        const results = await session.run({ [inputName]: tensor });
        const output = results[session.outputNames[0]];
        const probabilities = output.data;
        
        console.log('📊 MATLAB example prediction probabilities:', probabilities);
        
        let maxIndex = 0;
        let maxProb = 0;
        for (let i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                maxIndex = i;
            }
        }
        
        console.log(`🎯 MATLAB example prediction: ${obesityClasses[maxIndex]} (${(maxProb*100).toFixed(2)}%)`);
        return { predicted: obesityClasses[maxIndex], confidence: maxProb*100, probs: probabilities };
    } catch (error) {
        console.error('❌ MATLAB test error:', error);
        return null;
    }
}

// Make prediction
async function predict() {
    if (!session) {
        alert('Model not loaded yet. Please wait...');
        return;
    }
    
    // Collect form data
    const formData = {};
    const features = [
        'Gender', 'Age', 'Height', 'Weight', 
        'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP',
        'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
        'CALC', 'MTRANS'
    ];
    
    for (const feature of features) {
        const element = document.getElementById(feature);
        if (element) {
            formData[feature] = element.value;
        }
    }
    
    const predictBtn = document.getElementById('predictBtn');
    const originalText = predictBtn.innerHTML;
    predictBtn.disabled = true;
    predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Predicting...';
    
    try {
        console.log('🚀 Starting prediction...');
        const inputTensor = preprocessInput(formData);
        const inputName = session.inputNames[0];
        
        const startTime = performance.now();
        const results = await session.run({ [inputName]: inputTensor });
        const endTime = performance.now();
        
        console.log(`⏱️  Prediction time: ${(endTime - startTime).toFixed(2)} ms`);
        
        const output = results[session.outputNames[0]];
        const probabilities = output.data;
        console.log('📊 Probabilities:', probabilities);
        
        // Softmax sum
        const sum = probabilities.reduce((a, b) => a + b, 0);
        console.log(`🧮 Softmax sum: ${sum.toFixed(6)} (should be close to 1)`);
        
        // Find class with highest probability
        let maxIndex = 0;
        let maxProb = 0;
        for (let i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                maxIndex = i;
            }
        }
        
        const predictedClass = obesityClasses[maxIndex];
        const confidence = maxProb * 100;
        
        console.log(`🎯 Prediction: ${predictedClass} (${confidence.toFixed(2)}%)`);
        
        // Show results
        displayResults(predictedClass, confidence, probabilities);
        
    } catch (error) {
        console.error('❌ Prediction error:', error);
        alert(`Prediction error: ${error.message}`);
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerHTML = originalText;
    }
}

// Show results
function displayResults(predictedClass, confidence, probabilities) {
    document.getElementById('resultsCard').style.display = 'block';
    
    const englishClasses = {
        'Insufficient_Weight': 'Insufficient Weight',
        'Normal_Weight': 'Normal Weight',
        'Overweight_Level_I': 'Overweight Level I',
        'Overweight_Level_II': 'Overweight Level II',
        'Obesity_Type_I': 'Obesity Type I',
        'Obesity_Type_II': 'Obesity Type II',
        'Obesity_Type_III': 'Obesity Type III'
    };
    
    const classColors = {
        'Insufficient_Weight': 'info',
        'Normal_Weight': 'success',
        'Overweight_Level_I': 'warning',
        'Overweight_Level_II': 'warning',
        'Obesity_Type_I': 'danger',
        'Obesity_Type_II': 'danger',
        'Obesity_Type_III': 'danger'
    };
    
    // Prediction badge
    const predictionBadge = document.getElementById('predictionBadge');
    const color = classColors[predictedClass];
    predictionBadge.className = `badge bg-${color} fs-5 p-3`;
    predictionBadge.textContent = englishClasses[predictedClass] || predictedClass;
    
    // Confidence bar
    const confidenceBar = document.getElementById('confidenceBar');
    const confidencePercent = document.getElementById('confidencePercent');
    confidenceBar.className = `progress-bar bg-${color}`;
    confidenceBar.style.width = `${confidence}%`;
    confidencePercent.textContent = `${confidence.toFixed(1)}%`;
    
    // Show all probabilities
    const allProbsDiv = document.getElementById('allProbabilities');
    let html = '';
    
    for (let i = 0; i < probabilities.length; i++) {
        const percent = probabilities[i] * 100;
        const className = classColors[obesityClasses[i]];
        const englishName = englishClasses[obesityClasses[i]] || obesityClasses[i];
        
        html += `
            <div class="mb-2">
                <div class="d-flex justify-content-between">
                    <span class="badge bg-${className}" style="min-width: 180px;">${englishName}</span>
                    <span class="fw-bold">${percent.toFixed(2)}%</span>
                </div>
                <div class="progress" style="height: 8px;">
                    <div class="progress-bar bg-${className}" style="width: ${percent}%"></div>
                </div>
            </div>
        `;
    }
    
    allProbsDiv.innerHTML = html;
    document.getElementById('resultsCard').scrollIntoView({ behavior: 'smooth' });
}

// Reset form
function resetForm() {
    document.getElementById('obesityForm').reset();
    document.getElementById('resultsCard').style.display = 'none';
    
    // Default values
    document.getElementById('Age').value = '25';
    document.getElementById('Height').value = '170';
    document.getElementById('Weight').value = '70';
    document.getElementById('FCVC').value = '2.5';
    document.getElementById('NCP').value = '3';
    document.getElementById('CH2O').value = '2.0';
    document.getElementById('FAF').value = '1.5';
    document.getElementById('TUE').value = '1.0';
}

// On page load
document.addEventListener('DOMContentLoaded', function() {
    if (typeof ort === 'undefined') {
        document.getElementById('modelStatus').innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle"></i> ONNX Runtime failed to load!
            </div>
        `;
        return;
    }
    
    loadModel().then(() => {
        // Run MATLAB test after model loads
        setTimeout(() => testMATLABExample(), 1000);
    });
    
    // Form submit
    document.getElementById('obesityForm').addEventListener('submit', function(e) {
        e.preventDefault();
        predict();
    });
});

// Debug functions
window.debugModel = function() {
    console.log('=== DEBUG ===');
    console.log('Session:', session);
    console.log('Input names:', session?.inputNames);
    console.log('Output names:', session?.outputNames);
    console.log('Feature stats:', featureStats);
    console.log('Categorical mapping:', categoricalMapping);
    console.log('Obesity classes:', obesityClasses);
};

window.testMapping = function() {
    console.log('=== MAPPING TEST ===');
    console.log('CAEC mapping test:');
    console.log('  Always ->', categoricalMapping.CAEC.Always);    // 1
    console.log('  Frequently ->', categoricalMapping.CAEC.Frequently); // 2
    console.log('  no ->', categoricalMapping.CAEC.no);          // 3
    console.log('  Sometimes ->', categoricalMapping.CAEC.Sometimes); // 4
    
    console.log('\nNormalize test for CAEC=no (3):');
    const mean = featureStats.CAEC.mean; // 2.859308
    const std = featureStats.CAEC.std;   // 0.468543
    const normalized = (3 - mean) / std;
    console.log(`  (3 - ${mean}) / ${std} = ${normalized}`);
};