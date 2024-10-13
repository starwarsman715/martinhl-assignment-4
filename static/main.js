// static/main.js

document.getElementById('search-form').addEventListener('submit', function (event) {
    event.preventDefault();
    
    let query = document.getElementById('query').value;
    let resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';
    let chartCanvas = document.getElementById('similarity-chart');
    chartCanvas.style.display = 'none'; // Hide chart initially

    fetch('/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
            'query': query
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        if (data.error) {
            displayError(data.error);
        } else {
            displayResults(data);
            displayChart(data);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        displayError('An unexpected error occurred.');
    });
});

function displayResults(data) {
    let resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '<h2>Results</h2>';
    for (let i = 0; i < data.documents.length; i++) {
        let docDiv = document.createElement('div');
        let snippet = getSnippet(data.documents[i], data.indices[i]);
        let fromField = data.from_fields[i];
        let subjectField = data.subject_fields[i];
        
        docDiv.innerHTML = `
            <strong>Document ${data.indices[i]}</strong><br>
            <strong>From:</strong> ${fromField}<br>
            <strong>Subject:</strong> ${subjectField}<br>
            <p>${snippet}</p>
            <strong>Similarity: ${data.similarities[i]}</strong>
        `;
        resultsDiv.appendChild(docDiv);
    }
}

function getSnippet(documentText, docIndex) {
    // Extract a snippet from the document, e.g., the first 200 characters after headers
    // Assuming headers are at the beginning, find the first empty line
    let lines = documentText.split('\n');
    let contentStart = 0;
    for (let i = 0; i < lines.length; i++) {
        if (lines[i].trim() === '') {
            contentStart = i + 1;
            break;
        }
    }
    let content = lines.slice(contentStart).join(' ').trim();
    return content.length > 200 ? content.substring(0, 200) + '...' : content;
}

function displayChart(data) {
    const ctx = document.getElementById('similarity-chart').getContext('2d');
    const chartCanvas = document.getElementById('similarity-chart');
    chartCanvas.style.display = 'block'; // Show chart

    // Destroy existing chart if it exists to prevent duplication
    if (window.similarityChart) {
        window.similarityChart.destroy();
    }

    window.similarityChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.indices.map(index => `Doc ${index}`),
            datasets: [{
                label: 'Cosine Similarity',
                data: data.similarities,
                backgroundColor: 'rgba(54, 162, 235, 0.6)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Cosine Similarity of Top 5 Documents'
                }
            }
        }
    });
}

function displayError(message) {
    let resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = `<h2>Error</h2><p>${message}</p>`;
    let chartCanvas = document.getElementById('similarity-chart');
    chartCanvas.style.display = 'none'; // Hide chart
}
