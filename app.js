// F1 Race Prediction Application
class F1RacePredictor {
    constructor() {
        this.drivers = [
            {"abbr": "VER", "name": "Max Verstappen", "defaultTeam": "Red Bull Racing", "experience": 10},
            {"abbr": "HAM", "name": "Lewis Hamilton", "defaultTeam": "Mercedes", "experience": 9},
            {"abbr": "LEC", "name": "Charles Leclerc", "defaultTeam": "Ferrari", "experience": 8},
            {"abbr": "SAI", "name": "Carlos Sainz", "defaultTeam": "Ferrari", "experience": 7},
            {"abbr": "RUS", "name": "George Russell", "defaultTeam": "Mercedes", "experience": 6},
            {"abbr": "NOR", "name": "Lando Norris", "defaultTeam": "McLaren", "experience": 6},
            {"abbr": "PER", "name": "Sergio Perez", "defaultTeam": "Red Bull Racing", "experience": 7},
            {"abbr": "ALO", "name": "Fernando Alonso", "defaultTeam": "Aston Martin", "experience": 9},
            {"abbr": "PIA", "name": "Oscar Piastri", "defaultTeam": "McLaren", "experience": 4},
            {"abbr": "ALB", "name": "Alexander Albon", "defaultTeam": "Williams", "experience": 5}
        ];

        this.teams = [
            {"name": "Red Bull Racing", "performance": 10, "color": "#3671C6"},
            {"name": "Mercedes", "performance": 8, "color": "#6CD3BF"},
            {"name": "Ferrari", "performance": 7, "color": "#F91536"},
            {"name": "McLaren", "performance": 6, "color": "#F58020"},
            {"name": "Aston Martin", "performance": 5, "color": "#358C75"},
            {"name": "Williams", "performance": 4, "color": "#37BEDD"},
            {"name": "Alpine", "performance": 4, "color": "#2293D1"},
            {"name": "Haas F1 Team", "performance": 3, "color": "#B6BABD"},
            {"name": "AlphaTauri", "performance": 4, "color": "#5E8FAA"},
            {"name": "Alfa Romeo", "performance": 3, "color": "#C92D4B"}
        ];

        this.init();
    }

    init() {
        this.populateDriverTable();
        this.bindEvents();
    }

    populateDriverTable() {
        const tableBody = document.getElementById('driverTable');
        if (!tableBody) return;
        
        tableBody.innerHTML = '';

        this.drivers.forEach((driver, index) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td class="driver-abbr">${driver.abbr}</td>
                <td>
                    <select class="team-select form-control" id="team-${index}" data-driver="${driver.abbr}">
                        ${this.teams.map(team => 
                            `<option value="${team.name}" ${team.name === driver.defaultTeam ? 'selected' : ''}>${team.name}</option>`
                        ).join('')}
                    </select>
                </td>
                <td>
                    <input type="number" class="grid-input form-control" id="grid-${index}" 
                           min="1" max="20" value="${index + 1}" data-driver="${driver.abbr}">
                </td>
                <td>
                    <input type="number" class="time-input form-control" id="time-${index}" 
                           min="60" max="120" step="0.001" value="${(75 + Math.random() * 5).toFixed(3)}" 
                           data-driver="${driver.abbr}">
                </td>
            `;
            tableBody.appendChild(row);
        });
    }

    bindEvents() {
        const predictBtn = document.getElementById('predictBtn');
        const resetBtn = document.getElementById('resetBtn');
        
        if (predictBtn) {
            predictBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.handlePredictClick();
            });
        }
        
        if (resetBtn) {
            resetBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.resetForm();
            });
        }
    }

    handlePredictClick() {
        const predictBtn = document.getElementById('predictBtn');
        if (!predictBtn) return;

        // Show loading state
        const originalText = predictBtn.innerHTML;
        predictBtn.innerHTML = '<span>⏳</span> Analyzing Data...';
        predictBtn.disabled = true;

        // Validate form first
        if (!this.validateForm()) {
            predictBtn.innerHTML = originalText;
            predictBtn.disabled = false;
            return;
        }

        // Add delay for user experience
        setTimeout(() => {
            try {
                this.predictRace();
                predictBtn.innerHTML = originalText;
                predictBtn.disabled = false;
            } catch (error) {
                console.error('Prediction error:', error);
                alert('Error generating predictions. Please check your input data.');
                predictBtn.innerHTML = originalText;
                predictBtn.disabled = false;
            }
        }, 1000);
    }

    validateForm() {
        const gridInputs = document.querySelectorAll('.grid-input');
        const timeInputs = document.querySelectorAll('.time-input');
        let isValid = true;

        gridInputs.forEach(input => {
            const value = parseInt(input.value);
            if (!input.value || isNaN(value) || value < 1 || value > 20) {
                input.style.borderColor = '#DC143C';
                isValid = false;
            } else {
                input.style.borderColor = '';
            }
        });

        timeInputs.forEach(input => {
            const value = parseFloat(input.value);
            if (!input.value || isNaN(value) || value < 60 || value > 120) {
                input.style.borderColor = '#DC143C';
                isValid = false;
            } else {
                input.style.borderColor = '';
            }
        });

        if (!isValid) {
            alert('Please ensure all grid positions are between 1-20 and qualifying times are between 60-120 seconds.');
        }

        return isValid;
    }

    resetForm() {
        const raceNameInput = document.getElementById('raceName');
        if (raceNameInput) {
            raceNameInput.value = '';
        }
        
        this.populateDriverTable();
        
        const resultsSection = document.getElementById('resultsSection');
        if (resultsSection) {
            resultsSection.classList.add('hidden');
        }
    }

    collectDriverData() {
        const driverData = [];
        
        this.drivers.forEach((driver, index) => {
            const teamSelect = document.getElementById(`team-${index}`);
            const gridInput = document.getElementById(`grid-${index}`);
            const timeInput = document.getElementById(`time-${index}`);

            if (!teamSelect || !gridInput || !timeInput) {
                console.error(`Missing input elements for driver ${driver.abbr}`);
                return;
            }

            const selectedTeam = this.teams.find(team => team.name === teamSelect.value);
            if (!selectedTeam) {
                console.error(`Team not found: ${teamSelect.value}`);
                return;
            }
            
            driverData.push({
                abbr: driver.abbr,
                name: driver.name,
                team: teamSelect.value,
                teamPerformance: selectedTeam.performance,
                teamColor: selectedTeam.color,
                gridPosition: parseInt(gridInput.value) || 1,
                qualifyingTime: parseFloat(timeInput.value) || 75,
                experience: driver.experience
            });
        });

        return driverData;
    }

    calculatePredictionScore(driver) {
        // Normalize values for scoring (higher is better)
        const gridScore = Math.max(0, (21 - driver.gridPosition) / 20); // Invert grid position
        const teamScore = driver.teamPerformance / 10;
        const experienceScore = driver.experience / 10;
        
        // For qualifying time - normalize to 0-1 range
        const minTime = 70;
        const maxTime = 90;
        const normalizedTime = Math.max(0, Math.min(1, (maxTime - driver.qualifyingTime) / (maxTime - minTime)));
        const timeScore = normalizedTime;

        // Apply weights: Grid 40%, Team 30%, Experience 20%, Time 10%
        const totalScore = (
            gridScore * 0.4 +
            teamScore * 0.3 +
            experienceScore * 0.2 +
            timeScore * 0.1
        );

        return Math.max(0.1, Math.min(1, totalScore)); // Ensure score is between 0.1 and 1
    }

    predictRace() {
        console.log('Starting race prediction...');
        
        const driverData = this.collectDriverData();
        if (driverData.length === 0) {
            throw new Error('No driver data collected');
        }

        console.log('Driver data collected:', driverData);

        // Calculate prediction scores
        const predictions = driverData.map(driver => {
            const score = this.calculatePredictionScore(driver);
            return {
                ...driver,
                score: score
            };
        });

        // Sort by score (highest first)
        predictions.sort((a, b) => b.score - a.score);

        // Convert scores to probabilities
        const totalScore = predictions.reduce((sum, p) => sum + p.score, 0);
        predictions.forEach(prediction => {
            prediction.probability = totalScore > 0 ? (prediction.score / totalScore) * 100 : 10;
        });

        console.log('Predictions calculated:', predictions);

        // Display results
        this.displayResults(predictions);
    }

    displayResults(predictions) {
        console.log('Displaying results...');
        
        const resultsSection = document.getElementById('resultsSection');
        if (!resultsSection) {
            console.error('Results section not found');
            return;
        }

        // Show results section
        resultsSection.classList.remove('hidden');

        // Update podium
        this.updatePodium(predictions.slice(0, 3));

        // Update full rankings
        this.updateRankings(predictions);

        // Update confidence level
        this.updateConfidence(predictions);

        // Scroll to results with delay
        setTimeout(() => {
            resultsSection.scrollIntoView({ 
                behavior: 'smooth',
                block: 'start'
            });
        }, 200);
    }

    updatePodium(topThree) {
        console.log('Updating podium with:', topThree);
        
        // Update each podium position
        topThree.forEach((driver, index) => {
            const podiumElement = document.getElementById(`podiumP${index + 1}`);
            if (!podiumElement) return;
            
            const driverInfo = podiumElement.querySelector('.driver-info');
            const probability = podiumElement.querySelector('.probability');

            if (driverInfo && probability) {
                driverInfo.innerHTML = `
                    <div class="driver-name">${driver.name}</div>
                    <div class="driver-team">${driver.team}</div>
                `;
                probability.textContent = `${driver.probability.toFixed(1)}%`;
            }
        });
    }

    updateRankings(predictions) {
        console.log('Updating rankings...');
        
        const rankingsTable = document.getElementById('rankingsTable');
        if (!rankingsTable) {
            console.error('Rankings table not found');
            return;
        }
        
        rankingsTable.innerHTML = '';

        predictions.forEach((driver, index) => {
            const row = document.createElement('tr');
            
            row.innerHTML = `
                <td class="position-cell">${index + 1}</td>
                <td>
                    <div class="driver-name">${driver.name}</div>
                    <div class="driver-abbr">${driver.abbr}</div>
                </td>
                <td style="color: ${driver.teamColor}; font-weight: bold;">${driver.team}</td>
                <td>
                    <div>${driver.probability.toFixed(1)}%</div>
                    <div class="probability-bar">
                        <div class="probability-fill" style="width: ${Math.min(100, driver.probability * 2)}%"></div>
                    </div>
                </td>
                <td class="score-cell">${(driver.score * 100).toFixed(1)}</td>
            `;

            rankingsTable.appendChild(row);
        });
    }

    updateConfidence(predictions) {
        const scores = predictions.map(p => p.score);
        const maxScore = Math.max(...scores);
        const avgScore = scores.reduce((sum, score) => sum + score, 0) / scores.length;
        const scoreVariance = scores.reduce((sum, score) => sum + Math.pow(score - avgScore, 2), 0) / scores.length;
        
        // Calculate confidence (45-95%)
        const confidence = Math.min(95, Math.max(45, (maxScore * 70) + (50 - scoreVariance * 200)));
        
        const confidenceBar = document.getElementById('confidenceBar');
        const confidenceText = document.getElementById('confidenceText');
        
        if (confidenceBar && confidenceText) {
            // Animate confidence bar
            setTimeout(() => {
                confidenceBar.style.width = `${confidence}%`;
            }, 500);

            let confidenceLevel = 'Low';
            if (confidence > 70) confidenceLevel = 'High';
            else if (confidence > 55) confidenceLevel = 'Medium';

            confidenceText.textContent = `${confidenceLevel} confidence (${confidence.toFixed(1)}%) based on data quality and prediction consistency`;
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing F1 Race Predictor...');
    
    try {
        const predictor = new F1RacePredictor();
        console.log('F1 Race Predictor initialized successfully');
        
        // Store reference globally for debugging
        window.f1Predictor = predictor;
        
    } catch (error) {
        console.error('Error initializing F1 Race Predictor:', error);
    }

    // Add input validation listeners
    document.addEventListener('input', function(e) {
        if (e.target.classList.contains('grid-input')) {
            const value = parseInt(e.target.value);
            if (value < 1) e.target.value = 1;
            if (value > 20) e.target.value = 20;
        }
        
        if (e.target.classList.contains('time-input')) {
            const value = parseFloat(e.target.value);
            if (value < 60) e.target.value = 60;
            if (value > 120) e.target.value = 120;
        }
    });

    // Add hover effects for table rows
    document.addEventListener('mouseover', function(e) {
        const row = e.target.closest('tr');
        if (row && (row.closest('.qualifying-table') || row.closest('.rankings-table'))) {
            row.style.transform = 'scale(1.01)';
            row.style.transition = 'transform 0.2s ease';
        }
    });

    document.addEventListener('mouseout', function(e) {
        const row = e.target.closest('tr');
        if (row && (row.closest('.qualifying-table') || row.closest('.rankings-table'))) {
            row.style.transform = 'scale(1)';
        }
    });
});