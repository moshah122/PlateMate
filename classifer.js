const nutritionMap = {
    Apple: { calories: 95, protein: 0.5, carbs: 25, sugars: 19 },
    Banana: { calories: 105, protein: 1.3, carbs: 27, sugars: 14 },
    Chips: { calories: 150, protein: 2, carbs: 18, sugars: 1 },
    Cookies: { calories: 220, protein: 2, carbs: 29, sugars: 13 },
    Orange: { calories: 45, protein: 0.9, carbs: 11, sugars: 9 },
    Yogurt: { calories: 130, protein: 11, carbs: 15, sugars: 14 }
};

const classLabels = { 0: "Apple", 1: "Banana", 2: "Chips", 3: "Cookies", 4: "Orange", 5: "Yogurt"};

const imageInputs = [
    document.getElementById("input1"),
    document.getElementById("input2"),
    document.getElementById("input3"),
    document.getElementById("input4")
];

const imgElements = [
    document.getElementById("image1"),
    document.getElementById("image2"),
    document.getElementById("image3"),
    document.getElementById("image4")
];

const displays = [
    document.getElementById("display1"),
    document.getElementById("display2"),
    document.getElementById("display3"),
    document.getElementById("display4")
];

let classifier;

imageInputs.forEach((input, index) => {
    input.addEventListener("change", (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = () => {
                imgElements[index].src = reader.result;
                document.getElementById(`result${index + 1}`).textContent = "";
                displays[index].innerHTML = "";
            };
            reader.readAsDataURL(file);
        }
    });
});

async function classify() {
    classifier = await tflite.loadTFLiteModel("finalFoodFloat.tflite");

    let results = [];
    let aggCalories = 0;
    let aggProtein = 0;
    let aggCarbs = 0;
    let aggSugars = 0;

    for (let i = 0; i < imgElements.length; i++) {
        const img = imgElements[i];
        if (!img.src) continue;

        let input = tf.browser.fromPixels(img).cast('float32');
        input = tf.image.resizeBilinear(input, [224, 224]);
        input = tf.div(input, 255.0);
        input = tf.expandDims(input, 0);

        const result = await classifier.predict(input);
        const resultData = result.dataSync();
        const maxIndex = resultData.indexOf(Math.max(...resultData));
        const predictedClass = classLabels[maxIndex];
        const confidence = resultData[maxIndex];

        results.push({ index: i + 1, predictedClass, confidence });
        aggCalories += nutritionMap[predictedClass].calories;
        aggProtein += nutritionMap[predictedClass].protein;
        aggCarbs += nutritionMap[predictedClass].carbs;
        aggSugars += nutritionMap[predictedClass].sugars;

        displays[i].innerHTML = ` 
            <div>Calories: ${nutritionMap[predictedClass].calories}</div>
            <div>Protein: ${nutritionMap[predictedClass].protein}</div>
            <div>Carbs: ${nutritionMap[predictedClass].carbs}</div>
            <div>Sugars: ${nutritionMap[predictedClass].sugars}</div>
        `;
    }

    displayResults(results);
    document.getElementById("calCount").textContent = aggCalories.toFixed(2);
    document.getElementById("proteinCount").textContent = aggProtein.toFixed(2);
    document.getElementById("carbCount").textContent = aggCarbs.toFixed(2);
    document.getElementById("sugarCount").textContent = aggSugars.toFixed(2);
}

function displayResults(results) {
    results.forEach(result => {
        const resultText = `${result.predictedClass} (Confidence: ${(result.confidence * 100).toFixed(2)}%)`;
        const resultElement = document.getElementById(`result${result.index}`);
        resultElement.textContent = resultText;
        resultElement.style.color = "black";
    });
}
