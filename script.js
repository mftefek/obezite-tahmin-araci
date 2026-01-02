// --- 1. MATLAB UYUMLU VERİ SÖZLÜĞÜ (1-Based Indexing) ---
const veriSozlugu = {
    Gender: { 'Female': 1, 'Male': 2 },
    family_history: { 'no': 1, 'yes': 2 },
    FAVC: { 'no': 1, 'yes': 2 },
    CAEC: { 'Always': 1, 'Frequently': 2, 'Sometimes': 3, 'no': 4 },
    SMOKE: { 'no': 1, 'yes': 2 },
    SCC: { 'no': 1, 'yes': 2 },
    CALC: { 'Always': 1, 'Frequently': 2, 'Sometimes': 3, 'no': 4 },
    MTRANS: { 'Automobile': 1, 'Bike': 2, 'Motorbike': 3, 'Public_Transportation': 4, 'Walking': 5 }
};

// --- 2. MATLAB UYUMLU NORMALİZASYON DEĞERLERİ ---
const stats = {
    mean: [
        1.5059,  // Gender
        24.3126, // Age
        1.7017,  // Height
        86.5861, // Weight
        1.8176,  // family_history
        1.8839,  // FAVC
        2.4190,  // FCVC
        2.6856,  // NCP
        2.8593,  // CAEC
        1.0208,  // SMOKE
        2.0080,  // CH2O
        1.0455,  // SCC
        1.0103,  // FAF
        0.6579,  // TUE
        3.2686,  // CALC
        3.3652   // MTRANS
    ],
    std: [
        0.5001,  // Gender
        6.3460,  // Age
        0.0933,  // Height
        26.1912, // Weight
        0.3862,  // family_history
        0.3204,  // FAVC
        0.5339,  // FCVC
        0.7780,  // NCP
        0.4685,  // CAEC
        0.1429,  // SMOKE
        0.6130,  // CH2O
        0.2084,  // SCC
        0.8506,  // FAF
        0.6089,  // TUE
        0.5155,  // CALC
        1.2614   // MTRANS
    ]
};

// --- 3. SONUÇ ETİKETLERİ (ENGLISH) ---
const sonucEtiketleri = [
    "Insufficient Weight",       // Index 0
    "Normal Weight",             // Index 1
    "Obesity Type I",            // Index 2
    "Obesity Type II",           // Index 3
    "Obesity Type III",          // Index 4
    "Overweight Level I",        // Index 5
    "Overweight Level II"        // Index 6
];

// Helper: Argmax
function indexOfMax(arr) {
    if (arr.length === 0) return -1;
    let max = arr[0];
    let maxIndex = 0;
    for (let i = 1; i < arr.length; i++) {
        if (arr[i] > max) { maxIndex = i; max = arr[i]; }
    }
    return maxIndex;
}

async function tahminEt() {
    const sonucDiv = document.getElementById("sonuc");
    // Mesaj İngilizceye çevrildi
    sonucDiv.innerText = "Calculating..."; 
    sonucDiv.style.color = "blue";

    try {
        // Dosya ismi düzeltildi: obezite.onnx
        const session = await ort.InferenceSession.create('./obzeite.onnx');

        // 1. Ham verileri topla
        let rawValues = [
            veriSozlugu.Gender[document.getElementById("Gender").value],
            parseFloat(document.getElementById("Age").value),
            parseFloat(document.getElementById("Height").value),
            parseFloat(document.getElementById("Weight").value),
            veriSozlugu.family_history[document.getElementById("family_history").value],
            veriSozlugu.FAVC[document.getElementById("FAVC").value],
            parseFloat(document.getElementById("FCVC").value),
            parseFloat(document.getElementById("NCP").value),
            veriSozlugu.CAEC[document.getElementById("CAEC").value],
            veriSozlugu.SMOKE[document.getElementById("SMOKE").value],
            parseFloat(document.getElementById("CH2O").value),
            veriSozlugu.SCC[document.getElementById("SCC").value],
            parseFloat(document.getElementById("FAF").value),
            parseFloat(document.getElementById("TUE").value),
            veriSozlugu.CALC[document.getElementById("CALC").value],
            veriSozlugu.MTRANS[document.getElementById("MTRANS").value]
        ];

        // 2. Normalizasyon
        let normalizedValues = rawValues.map((val, index) => {
            return (val - stats.mean[index]) / stats.std[index];
        });

        console.log("Raw Values:", rawValues);
        console.log("Normalized Values:", normalizedValues);

        // 3. Tensor Oluşturma [1, 1, 16, 1]
        const tensor = new ort.Tensor('float32', Float32Array.from(normalizedValues), [1, 1, 16, 1]);

        // 4. Çalıştır
        const inputName = session.inputNames[0];
        const feeds = {};
        feeds[inputName] = tensor;

        const results = await session.run(feeds);
        const outputName = session.outputNames[0];
        const outputTensor = results[outputName];

        // 5. Sonuç
        const tahminIndeksi = indexOfMax(outputTensor.data);
        // Bilinmiyor -> Unknown
        const tahminMetni = sonucEtiketleri[tahminIndeksi] || "Unknown"; 

        // Sonuç -> Result
        sonucDiv.innerText = "Result: " + tahminMetni;
        sonucDiv.style.color = "green";

    } catch (e) {
        // Hata -> Error
        sonucDiv.innerText = "Error: " + e.message;
        sonucDiv.style.color = "red";
        console.error(e);
    }
}