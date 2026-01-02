// ONNX model session
let session = null;

// Özellik istatistikleri - MATLAB normalize() değerleri
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

// DÜZELTİLMİŞ: Alfabetik sıraya göre kategorik mapping
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

// Obezite sınıfları
const obesityClasses = [
    'Insufficient_Weight',
    'Normal_Weight',
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III'
];

// Model yükleme
async function loadModel() {
    try {
        console.log('Model yükleniyor...');
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
        session = await ort.InferenceSession.create('./obzeite.onnx', {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
        
        console.log('✅ Model yüklendi! Input:', session.inputNames[0], 'Output:', session.outputNames[0]);
        document.getElementById('modelStatus').style.display = 'none';
        document.getElementById('modelLoaded').style.display = 'block';
        
        return true;
    } catch (error) {
        console.error('❌ Model yükleme hatası:', error);
        document.getElementById('modelStatus').innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle"></i> Model yüklenemedi: ${error.message}
            </div>
        `;
        return false;
    }
}

// MATLAB reshape ile uyumlu veri hazırlama
function preprocessInput(formData) {
    // MATLAB'deki sırayla özellikler
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
        
        // Boyu metreye çevir
        if (feature === 'Height') {
            value = value / 100;
        }
        
        let numericValue;
        
        // Kategorik değişkenleri sayısallaştır
        if (categoricalMapping[feature]) {
            const mapping = categoricalMapping[feature];
            if (mapping[value] === undefined) {
                console.warn(`⚠️ Bilinmeyen değer: ${feature}=${value}`);
                // Alfabetik ilk değeri kullan
                const firstKey = Object.keys(mapping)[0];
                numericValue = mapping[firstKey];
            } else {
                numericValue = mapping[value];
            }
        } else {
            numericValue = parseFloat(value);
        }
        
        // MATLAB normalize() ile aynı işlem
        const mean = featureStats[feature].mean;
        const std = featureStats[feature].std;
        const normalized = (numericValue - mean) / std;
        
        processed.push(normalized);
    }
    
    console.log('📋 Ham değerler:', rawValues);
    console.log('🔢 Normalize edilmiş değerler:', processed.map(v => v.toFixed(4)));
    
    // MATLAB'deki reshape: XTrain = reshape(XTrain', [size(X_numeric,2), 1, 1, size(XTrain,1)]);
    // JavaScript'te: [batch, channels, height, width] = [1, 1, 16, 1]
    const floatArray = new Float32Array(processed);
    const tensor = new ort.Tensor('float32', floatArray, [1, 1, 16, 1]);
    
    console.log('🎯 Tensör şekli:', tensor.dims);
    return tensor;
}

// Test için MATLAB'deki bir örneği simüle et
async function testMATLABExample() {
    console.log('🧪 MATLAB örneği test ediliyor...');
    
    // MATLAB'deki normalize edilmiş değerleri kullan
    // Örnek: Ortalama değerler (tüm özellikler 0 olmalı)
    const testData = {
        'Gender': 'Female', // 1
        'Age': 24.3126, // normalize: (24.3126 - 24.3126) / 6.345968 = 0
        'Height': 1.701677, // normalize: 0
        'Weight': 86.586058, // normalize: 0
        'family_history_with_overweight': 'yes', // 2
        'FAVC': 'yes', // 2
        'FCVC': 2.419043, // 0
        'NCP': 2.685628, // 0
        'CAEC': 'no', // 3 (alfabetik sıra)
        'SMOKE': 'no', // 1
        'CH2O': 2.008011, // 0
        'SCC': 'no', // 1
        'FAF': 1.010298, // 0
        'TUE': 0.657866, // 0
        'CALC': 'no', // 3 (alfabetik sıra)
        'MTRANS': 'Public_Transportation' // 4
    };
    
    // Kategorik değerlerin normalize edilmiş hali:
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
        
        console.log('📊 MATLAB örneği tahmin olasılıkları:', probabilities);
        
        let maxIndex = 0;
        let maxProb = 0;
        for (let i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                maxIndex = i;
            }
        }
        
        console.log(`🎯 MATLAB örneği tahmini: ${obesityClasses[maxIndex]} (${(maxProb*100).toFixed(2)}%)`);
        return { predicted: obesityClasses[maxIndex], confidence: maxProb*100, probs: probabilities };
    } catch (error) {
        console.error('❌ MATLAB test hatası:', error);
        return null;
    }
}

// Tahmin yap
async function predict() {
    if (!session) {
        alert('Model henüz yüklenmedi. Lütfen bekleyin...');
        return;
    }
    
    // Form verilerini topla
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
    predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Tahmin Yapılıyor...';
    
    try {
        console.log('🚀 Tahmin başlatılıyor...');
        const inputTensor = preprocessInput(formData);
        const inputName = session.inputNames[0];
        
        const startTime = performance.now();
        const results = await session.run({ [inputName]: inputTensor });
        const endTime = performance.now();
        
        console.log(`⏱️  Tahmin süresi: ${(endTime - startTime).toFixed(2)} ms`);
        
        const output = results[session.outputNames[0]];
        const probabilities = output.data;
        console.log('📊 Olasılıklar:', probabilities);
        
        // Softmax toplamı
        const sum = probabilities.reduce((a, b) => a + b, 0);
        console.log(`🧮 Softmax toplamı: ${sum.toFixed(6)} (1'e yakın olmalı)`);
        
        // En yüksek olasılıklı sınıfı bul
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
        
        console.log(`🎯 Tahmin: ${predictedClass} (${confidence.toFixed(2)}%)`);
        
        // Sonuçları göster
        displayResults(predictedClass, confidence, probabilities);
        
    } catch (error) {
        console.error('❌ Tahmin hatası:', error);
        alert(`Tahmin hatası: ${error.message}`);
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerHTML = originalText;
    }
}

// Sonuçları göster
function displayResults(predictedClass, confidence, probabilities) {
    document.getElementById('resultsCard').style.display = 'block';
    
    const turkishClasses = {
        'Insufficient_Weight': 'Yetersiz Kilolu',
        'Normal_Weight': 'Normal Kilolu',
        'Overweight_Level_I': 'Fazla Kilolu Seviye I',
        'Overweight_Level_II': 'Fazla Kilolu Seviye II',
        'Obesity_Type_I': 'Obezite Tip I',
        'Obesity_Type_II': 'Obezite Tip II',
        'Obesity_Type_III': 'Obezite Tip III'
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
    
    // Tahmin rozeti
    const predictionBadge = document.getElementById('predictionBadge');
    const color = classColors[predictedClass];
    predictionBadge.className = `badge bg-${color} fs-5 p-3`;
    predictionBadge.textContent = turkishClasses[predictedClass] || predictedClass;
    
    // Güven çubuğu
    const confidenceBar = document.getElementById('confidenceBar');
    const confidencePercent = document.getElementById('confidencePercent');
    confidenceBar.className = `progress-bar bg-${color}`;
    confidenceBar.style.width = `${confidence}%`;
    confidencePercent.textContent = `${confidence.toFixed(1)}%`;
    
    // Tüm olasılıkları göster
    const allProbsDiv = document.getElementById('allProbabilities');
    let html = '';
    
    for (let i = 0; i < probabilities.length; i++) {
        const percent = probabilities[i] * 100;
        const className = classColors[obesityClasses[i]];
        const turkishName = turkishClasses[obesityClasses[i]] || obesityClasses[i];
        
        html += `
            <div class="mb-2">
                <div class="d-flex justify-content-between">
                    <span class="badge bg-${className}" style="min-width: 180px;">${turkishName}</span>
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

// BMI hesaplama
function calculateBMI() {
    const height = parseFloat(document.getElementById('bmiHeight').value) / 100;
    const weight = parseFloat(document.getElementById('bmiWeight').value);
    
    if (height <= 0 || weight <= 0) {
        document.getElementById('bmiResult').innerHTML = '<div class="alert alert-warning">Geçerli değerler giriniz.</div>';
        return;
    }
    
    const bmi = weight / (height * height);
    let category, color;
    
    if (bmi < 18.5) { category = 'Zayıf'; color = 'info'; }
    else if (bmi < 25) { category = 'Normal'; color = 'success'; }
    else if (bmi < 30) { category = 'Fazla Kilolu'; color = 'warning'; }
    else { category = 'Obez'; color = 'danger'; }
    
    document.getElementById('bmiResult').innerHTML = `
        <div class="alert alert-${color}">
            <h6>BMI Sonucu:</h6>
            <p class="mb-1"><strong>Değer:</strong> ${bmi.toFixed(1)}</p>
            <p class="mb-0"><strong>Kategori:</strong> ${category}</p>
        </div>
    `;
}

// Formu sıfırlama
function resetForm() {
    document.getElementById('obesityForm').reset();
    document.getElementById('resultsCard').style.display = 'none';
    
    // Varsayılan değerler
    document.getElementById('Age').value = '25';
    document.getElementById('Height').value = '170';
    document.getElementById('Weight').value = '70';
    document.getElementById('FCVC').value = '2.5';
    document.getElementById('NCP').value = '3';
    document.getElementById('CH2O').value = '2.0';
    document.getElementById('FAF').value = '1.5';
    document.getElementById('TUE').value = '1.0';
}

// Sayfa yüklendiğinde
document.addEventListener('DOMContentLoaded', function() {
    if (typeof ort === 'undefined') {
        document.getElementById('modelStatus').innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle"></i> ONNX Runtime yüklenemedi!
            </div>
        `;
        return;
    }
    
    loadModel().then(() => {
        // Model yüklendikten sonra MATLAB testi yap
        setTimeout(() => testMATLABExample(), 1000);
    });
    
    // Form submit
    document.getElementById('obesityForm').addEventListener('submit', function(e) {
        e.preventDefault();
        predict();
    });
    
    // BMI sync
    document.getElementById('bmiHeight').addEventListener('input', function() {
        document.getElementById('Height').value = this.value;
    });
    document.getElementById('bmiWeight').addEventListener('input', function() {
        document.getElementById('Weight').value = this.value;
    });
    document.getElementById('Height').addEventListener('input', function() {
        document.getElementById('bmiHeight').value = this.value;
    });
    document.getElementById('Weight').addEventListener('input', function() {
        document.getElementById('bmiWeight').value = this.value;
    });
});

// Debug fonksiyonları
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