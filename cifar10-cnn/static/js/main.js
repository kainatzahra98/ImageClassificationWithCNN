/* ═══════════════════════════════════════════════════════════
   CIFAR-10 CNN Classifier — Dashboard JavaScript
   ═══════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
    initParticles();
    initNavbar();
    initStatsAnimation();
    initUploader();
    initRevealAnimations();
    initPerfBars();
});

/* ── Background Particles ─────────────────────────────── */
function initParticles() {
    const container = document.getElementById('bgParticles');
    const colors = ['#6c5ce7', '#a855f7', '#00d2ff', '#00e676'];
    
    for (let i = 0; i < 30; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        const size = Math.random() * 4 + 2;
        const color = colors[Math.floor(Math.random() * colors.length)];
        
        particle.style.cssText = `
            width: ${size}px;
            height: ${size}px;
            left: ${Math.random() * 100}%;
            background: ${color};
            --duration: ${Math.random() * 15 + 10}s;
            --delay: ${Math.random() * 10}s;
            --opacity: ${Math.random() * 0.4 + 0.1};
        `;
        
        container.appendChild(particle);
    }
}

/* ── Navbar ───────────────────────────────────────────── */
function initNavbar() {
    const navbar = document.getElementById('navbar');
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = ['hero', 'predict', 'results', 'architecture'];
    
    // Scroll effect
    window.addEventListener('scroll', () => {
        navbar.classList.toggle('scrolled', window.scrollY > 50);
        
        // Active section tracking
        const scrollPos = window.scrollY + 200;
        for (let i = sections.length - 1; i >= 0; i--) {
            const section = document.getElementById(sections[i]);
            if (section && section.offsetTop <= scrollPos) {
                navLinks.forEach(link => link.classList.remove('active'));
                const activeLink = document.querySelector(`[data-section="${sections[i]}"]`);
                if (activeLink) activeLink.classList.add('active');
                break;
            }
        }
    });
    
    // Smooth scroll for nav links
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('data-section');
            const target = document.getElementById(targetId);
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
}

/* ── Animated Stats ───────────────────────────────────── */
function initStatsAnimation() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateValue('animatedAccuracy', 0, 86.8, 2000, 1);
                animateValue('animatedParams', 0, 1.2, 2000, 1);
                animateValue('animatedEpochs', 0, 30, 1500, 0);
                observer.disconnect();
            }
        });
    }, { threshold: 0.3 });
    
    const heroStats = document.querySelector('.hero-stats');
    if (heroStats) observer.observe(heroStats);
}

function animateValue(elementId, start, end, duration, decimals) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    const startTime = performance.now();
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = start + (end - start) * eased;
        
        element.textContent = current.toFixed(decimals);
        
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    
    requestAnimationFrame(update);
}

/* ── File Upload & Prediction ─────────────────────────── */
function initUploader() {
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const uploadContent = document.getElementById('uploadContent');
    const uploadPreview = document.getElementById('uploadPreview');
    const previewImage = document.getElementById('previewImage');
    const btnRemove = document.getElementById('btnRemove');
    const btnPredict = document.getElementById('btnPredict');
    const btnPredictText = document.getElementById('btnPredictText');
    const btnSpinner = document.getElementById('btnSpinner');
    const resultsPanel = document.getElementById('resultsPanel');
    
    let selectedFile = null;
    
    // Click to upload
    uploadZone.addEventListener('click', (e) => {
        if (e.target !== btnRemove && !btnRemove.contains(e.target)) {
            fileInput.click();
        }
    });
    
    // Drag and drop
    ['dragenter', 'dragover'].forEach(event => {
        uploadZone.addEventListener(event, (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });
    });
    
    ['dragleave', 'drop'].forEach(event => {
        uploadZone.addEventListener(event, (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
        });
    });
    
    uploadZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.startsWith('image/')) {
            handleFile(files[0]);
        }
    });
    
    // File input change
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });
    
    // Handle file selection
    function handleFile(file) {
        selectedFile = file;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadContent.style.display = 'none';
            uploadPreview.style.display = 'flex';
            btnPredict.disabled = false;
        };
        reader.readAsDataURL(file);
        
        // Hide previous results
        resultsPanel.style.display = 'none';
    }
    
    // Remove image
    btnRemove.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedFile = null;
        fileInput.value = '';
        uploadContent.style.display = 'block';
        uploadPreview.style.display = 'none';
        btnPredict.disabled = true;
        resultsPanel.style.display = 'none';
    });
    
    // Predict
    btnPredict.addEventListener('click', async () => {
        if (!selectedFile) return;
        
        // Loading state
        btnPredict.disabled = true;
        btnSpinner.style.display = 'inline-block';
        btnPredictText.textContent = 'Analyzing...';
        resultsPanel.style.display = 'none';
        
        try {
            const formData = new FormData();
            formData.append('image', selectedFile);
            
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                displayResults(data);
            } else {
                alert('Prediction failed: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            alert('Error connecting to server: ' + error.message);
        } finally {
            btnPredict.disabled = false;
            btnSpinner.style.display = 'none';
            btnPredictText.textContent = '🧠 Classify Image';
        }
    });
}

/* ── Display Prediction Results ───────────────────────── */
function displayResults(data) {
    const resultsPanel = document.getElementById('resultsPanel');
    const resultEmoji = document.getElementById('resultEmoji');
    const resultClass = document.getElementById('resultClass');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceText = document.getElementById('confidenceText');
    const probabilityGrid = document.getElementById('probabilityGrid');
    
    // Main prediction
    resultEmoji.textContent = data.emoji;
    resultClass.textContent = data.prediction;
    confidenceBar.style.width = '0%';
    confidenceText.textContent = `${data.confidence}% confidence`;
    
    // Animate confidence bar
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            confidenceBar.style.width = `${data.confidence}%`;
        });
    });
    
    // Probability breakdown
    probabilityGrid.innerHTML = '';
    data.all_probabilities.forEach((item, index) => {
        const isTop = index === 0;
        const div = document.createElement('div');
        div.className = `prob-item ${isTop ? 'top-result' : ''}`;
        div.innerHTML = `
            <span class="prob-emoji">${item.emoji}</span>
            <span class="prob-name">${item.class}</span>
            <div class="prob-bar-container">
                <div class="prob-bar" style="width: 0%;"></div>
            </div>
            <span class="prob-value">${item.probability.toFixed(1)}%</span>
        `;
        probabilityGrid.appendChild(div);
        
        // Animate bars
        requestAnimationFrame(() => {
            setTimeout(() => {
                div.querySelector('.prob-bar').style.width = `${item.probability}%`;
            }, index * 60);
        });
    });
    
    // Show results panel
    resultsPanel.style.display = 'block';
    
    // Scroll to results
    setTimeout(() => {
        resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 200);
}

/* ── Reveal Animations (Intersection Observer) ────────── */
function initRevealAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });
    
    // Add reveal class to sections
    const elements = document.querySelectorAll(
        '.result-card, .arch-node, .tech-card, .section-header'
    );
    elements.forEach(el => {
        el.classList.add('reveal');
        observer.observe(el);
    });
}

/* ── Per-Class Performance Bar Animation ──────────────── */
function initPerfBars() {
    const perfSection = document.getElementById('classPerformance');
    if (!perfSection) return;
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const bars = perfSection.querySelectorAll('.perf-bar-fill');
                bars.forEach(bar => bar.classList.add('animate'));
                observer.disconnect();
            }
        });
    }, { threshold: 0.2 });
    
    observer.observe(perfSection);
}
