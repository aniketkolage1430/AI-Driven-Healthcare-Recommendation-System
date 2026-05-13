let probabilityChart;
let modelChart;

const form = document.getElementById("predictionForm");
const symptomsInput = document.getElementById("symptomsInput");
const typingStatus = document.getElementById("typingStatus");
const predictionCards = document.getElementById("predictionCards");
const recommendationCards = document.getElementById("recommendationCards");
const primaryDisease = document.getElementById("primaryDisease");
const diseaseDescription = document.getElementById("diseaseDescription");
const matchedSymptoms = document.getElementById("matchedSymptoms");
const confidenceText = document.getElementById("confidenceText");
const confidenceRing = document.getElementById("confidenceRing");
const chatWindow = document.getElementById("chatWindow");
const historyList = document.getElementById("historyList");

document.addEventListener("DOMContentLoaded", () => {
    drawModelChart();
    renderValidationSummary();
    renderHistory();
});

document.getElementById("darkModeBtn").addEventListener("click", () => {
    document.body.classList.toggle("dark-mode");
    localStorage.setItem("medai_dark", document.body.classList.contains("dark-mode"));
});

if (localStorage.getItem("medai_dark") === "true") {
    document.body.classList.add("dark-mode");
}

document.querySelectorAll(".quick-chips button").forEach((button) => {
    button.addEventListener("click", () => {
        symptomsInput.value = button.dataset.symptoms;
        symptomsInput.focus();
    });
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const symptoms = symptomsInput.value.trim();
    if (!symptoms) {
        showBotMessage("Please enter symptoms first, for example: fever, chills, sweating.");
        return;
    }

    setLoading(true);
    showUserMessage(symptoms);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ symptoms }),
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || "Unable to process symptoms.");
        }
        renderPrediction(data);
        saveHistory(data);
        showBotMessage(`I found ${data.primary_disease} as the strongest match with ${data.confidence}% confidence. Please review the recommendations and consult a doctor for confirmation.`);
    } catch (error) {
        showBotMessage(error.message);
    } finally {
        setLoading(false);
    }
});

document.getElementById("voiceBtn").addEventListener("click", () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        showBotMessage("Voice input is not supported in this browser. Chrome or Edge works best.");
        return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = "en-IN";
    recognition.interimResults = false;
    recognition.start();
    showBotMessage("Listening for symptoms...");

    recognition.onresult = (event) => {
        symptomsInput.value = event.results[0][0].transcript;
    };
    recognition.onerror = () => showBotMessage("Voice input stopped. Please try again or type symptoms manually.");
});

document.getElementById("exportBtn").addEventListener("click", () => {
    window.print();
});

document.getElementById("clearHistoryBtn").addEventListener("click", () => {
    localStorage.removeItem("medai_history");
    renderHistory();
});

function setLoading(isLoading) {
    typingStatus.classList.toggle("d-none", !isLoading);
}

function drawModelChart() {
    const ctx = document.getElementById("modelChart");
    const labels = Object.keys(window.MODEL_SCORES || {});
    const values = Object.values(window.MODEL_SCORES || {}).map((score) => Math.round(score * 100));

    modelChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [{
                label: "Accuracy %",
                data: values,
                backgroundColor: ["#2563eb", "#16a34a", "#f59e0b", "#7c3aed", "#ef4444"],
                borderRadius: 6,
            }],
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true, max: 100 } },
        },
    });
}

function renderValidationSummary() {
    const summary = window.VALIDATION_SUMMARY || {};
    document.getElementById("precisionMetric").textContent = `${summary.weighted_precision || 0}%`;
    document.getElementById("recallMetric").textContent = `${summary.weighted_recall || 0}%`;
    document.getElementById("f1Metric").textContent = `${summary.weighted_f1 || 0}%`;

    const labels = summary.class_preview || [];
    const rows = summary.confusion_preview || [];
    const header = `<tr><th>Disease</th>${labels.map((label) => `<th>${label.slice(0, 8)}</th>`).join("")}</tr>`;
    const body = rows.map((row, index) => `
        <tr>
            <th>${(labels[index] || `Class ${index + 1}`).slice(0, 14)}</th>
            ${row.slice(0, 8).map((value) => `<td>${value}</td>`).join("")}
        </tr>
    `).join("");

    document.getElementById("matrixPreview").innerHTML = `<table>${header}${body}</table>`;
}

function renderPrediction(data) {
    predictionCards.classList.remove("empty-state");
    predictionCards.innerHTML = data.predictions.map((item, index) => `
        <div class="prediction-card" style="animation-delay:${index * 0.05}s">
            <header>
                <span>${index + 1}. ${item.disease}</span>
                <span>${item.percentage}%</span>
            </header>
            <div class="progress">
                <div class="progress-bar bg-success" style="width:${item.percentage}%"></div>
            </div>
        </div>
    `).join("");

    primaryDisease.textContent = data.primary_disease;
    diseaseDescription.textContent = data.recommendation.description;
    matchedSymptoms.textContent = `Matched: ${data.matched_symptoms.join(", ")}`;
    confidenceText.textContent = `${data.confidence}%`;
    confidenceRing.textContent = `${Math.round(data.confidence)}%`;

    renderProbabilityChart(data.predictions);
    renderRecommendations(data.recommendation);
}

function renderProbabilityChart(predictions) {
    const ctx = document.getElementById("probabilityChart");
    if (probabilityChart) {
        probabilityChart.destroy();
    }

    probabilityChart = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: predictions.map((item) => item.disease),
            datasets: [{
                data: predictions.map((item) => item.percentage),
                backgroundColor: ["#2563eb", "#16a34a", "#f59e0b", "#7c3aed", "#ef4444"],
                borderWidth: 0,
            }],
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: "bottom" },
            },
        },
    });
}

function renderRecommendations(recommendation) {
    const cards = [
        { title: "Precautions", icon: "fa-shield-heart", items: recommendation.precautions },
        { title: "Medicines", icon: "fa-pills", items: recommendation.medications },
        { title: "Diet Plan", icon: "fa-bowl-food", items: recommendation.diets },
        { title: "Workout", icon: "fa-person-walking", items: recommendation.workouts },
        { title: "AI Tips", icon: "fa-lightbulb", items: recommendation.tips },
    ];

    recommendationCards.innerHTML = cards.map((card) => `
        <div class="recommendation-card">
            <h4><i class="fa-solid ${card.icon}"></i> ${card.title}</h4>
            <ul>${card.items.map((item) => `<li>${item}</li>`).join("")}</ul>
        </div>
    `).join("");
}

function showUserMessage(message) {
    const bubble = document.createElement("div");
    bubble.className = "user-message";
    bubble.textContent = message;
    chatWindow.appendChild(bubble);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function showBotMessage(message) {
    const bubble = document.createElement("div");
    bubble.className = "bot-message";
    bubble.textContent = "";
    chatWindow.appendChild(bubble);
    typeText(bubble, message);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function typeText(element, message) {
    let index = 0;
    const timer = setInterval(() => {
        element.textContent += message.charAt(index);
        index += 1;
        if (index >= message.length) {
            clearInterval(timer);
        }
    }, 15);
}

function saveHistory(data) {
    const history = JSON.parse(localStorage.getItem("medai_history") || "[]");
    history.unshift({
        symptoms: data.input,
        disease: data.primary_disease,
        confidence: data.confidence,
        time: new Date().toLocaleString(),
    });
    localStorage.setItem("medai_history", JSON.stringify(history.slice(0, 8)));
    renderHistory();
}

function renderHistory() {
    const history = JSON.parse(localStorage.getItem("medai_history") || "[]");
    if (!history.length) {
        historyList.innerHTML = `<div class="history-item">No prediction history yet.</div>`;
        return;
    }

    historyList.innerHTML = history.map((item) => `
        <div class="history-item">
            <strong>${item.disease} - ${item.confidence}%</strong>
            <span>${item.symptoms}</span>
            <small class="d-block text-secondary">${item.time}</small>
        </div>
    `).join("");
}
