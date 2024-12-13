//the nutrition map for each one of the 6 categories of food items and their relevant info
const nutritionMap = {
    Apple: { calories: 95, protein: 0.5, carbs: 25, sugars: 19 },
    Banana: { calories: 105, protein: 1.3, carbs: 27, sugars: 14 },
    Chips: { calories: 150, protein: 2, carbs: 18, sugars: 1 },
    Cookies: { calories: 220, protein: 2, carbs: 29, sugars: 13 },
    Orange: { calories: 45, protein: 0.9, carbs: 11, sugars: 9 },
    Yogurt: { calories: 130, protein: 11, carbs: 15, sugars: 14 }
};

//a dictionary mapping the id of each item to the name
const classes = { 0: "Apple", 1: "Banana", 2: "Chips", 3: "Cookies", 4: "Orange", 5: "Yogurt"};

//a list of all the file inputs
const inputs = [
    document.getElementById("input1"),
    document.getElementById("input2"),
    document.getElementById("input3"),
    document.getElementById("input4")
];

//a list of all of the images inserted
const images = [
    document.getElementById("image1"),
    document.getElementById("image2"),
    document.getElementById("image3"),
    document.getElementById("image4")
];

//a list of the individual displays for each input
const displays = [
    document.getElementById("display1"),
    document.getElementById("display2"),
    document.getElementById("display3"),
    document.getElementById("display4")
];

let classifier; //this will be the model to store the tflite model into

//reads in the files for each the inputs and sets the result for each input
inputs.forEach((input, index) => {
    input.addEventListener("change", (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = () => {
                images[index].src = reader.result;
                document.getElementById(`result${index + 1}`).textContent = "";
                displays[index].innerHTML = "";
            };
            reader.readAsDataURL(file);
        }
    });
});

//classify function
async function classify() {
    //here we will upload the tflite model
    classifier = await tflite.loadTFLiteModel("finalFoodFloat.tflite");

    //initializes each of the aggregate values and the results list
    let results = [];
    let aggCalories = 0;
    let aggProtein = 0;
    let aggCarbs = 0;
    let aggSugars = 0;

    //for each one of the images, read it in and classify
    for (let i = 0; i < images.length; i++) {
        const img = images[i];
        if (!img.src) continue;

        //handles the preprocessing stage - sets the input to float32 and resizes, normalizes, and expands the tensor
        let input = tf.browser.fromPixels(img).cast('float32');
        input = tf.image.resizeBilinear(input, [224, 224]);
        input = tf.div(input, 255.0);
        input = tf.expandDims(input, 0);

        //predicts the result and then gets the array
        const result = await classifier.predict(input);
        const array = result.dataSync();
        const index = array.indexOf(Math.max(...array));

        //gets the prediction and confidence level for the class
        const predicted = classes[index];
        const confidence = array[index];

        //pushes this result to the list, including the class and confidence
        results.push({ index: i + 1, predicted, confidence });

        //adds up the totals for each one of the variables
        aggCalories += nutritionMap[predicted].calories;
        aggProtein += nutritionMap[predicted].protein;
        aggCarbs += nutritionMap[predicted].carbs;
        aggSugars += nutritionMap[predicted].sugars;

        //sets the innerHTML as specified in the html doc with the division
        displays[i].innerHTML = ` 
            <div>Calories: ${nutritionMap[predicted].calories}</div>
            <div>Protein: ${nutritionMap[predicted].protein}</div>
            <div>Carbs: ${nutritionMap[predicted].carbs}</div>
            <div>Sugars: ${nutritionMap[predicted].sugars}</div>
        `;
    }

    //displays the individual results
    displayResults(results);

    //set the texts for each one of the elements to their aggregate values
    document.getElementById("calCount").textContent = aggCalories.toFixed(2);
    document.getElementById("proteinCount").textContent = aggProtein.toFixed(2);
    document.getElementById("carbCount").textContent = aggCarbs.toFixed(2);
    document.getElementById("sugarCount").textContent = aggSugars.toFixed(2);
}

//function to display the individual results on the screen
function displayResults(results) {
    results.forEach(result => {
        const text = `${result.predicted} (Confidence: ${(result.confidence * 100).toFixed(2)}%)`;
        const resultCon = document.getElementById(`result${result.index}`);
        resultCon.textContent = text;
        resultCon.style.color = "black";
    });
}
