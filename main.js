/* AgriSmart AI — Enhanced Frontend JS */

// ── DEMO VALUES ──────────────────────────────────────────────────────────────
const DEMO_VALUES = {
  N: 90, P: 42, K: 43,
  temperature: 20.87, humidity: 82, ph: 6.5, rainfall: 202.9
};

// ── CHART INSTANCES ──────────────────────────────────────────────────────────
let importanceChart = null;
let cropChart = null;

// ── ELEMENTS ──────────────────────────────────────────────────────────────────
const form          = document.getElementById('prediction-form');
const submitBtn     = document.getElementById('submit-btn');
const fillDemoBtn   = document.getElementById('fill-demo');

const stateIdle     = document.getElementById('state-idle');
const stateLoading  = document.getElementById('state-loading');
const stateResult   = document.getElementById('state-result');

const resultEmoji   = document.getElementById('result-emoji');
const resultCrop    = document.getElementById('result-crop');
const confidencePct = document.getElementById('confidence-pct');
const confidenceFill= document.getElementById('confidence-fill');
const cropTip       = document.getElementById('crop-tip');
const metaSeason    = document.getElementById('meta-season');
const metaWater     = document.getElementById('meta-water');
const locationInput = document.getElementById('location');
const fetchWeatherBtn = document.getElementById('fetch-weather');
const weatherMessage = document.getElementById('weather-message');

const altCard       = document.getElementById('alt-card');
const chartCard     = document.getElementById('chart-card');
const top5List      = document.getElementById('top5-list');

// ── TABS ──────────────────────────────────────────────────────────────────────
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-section').forEach(s => s.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + tab).classList.add('active');

    if (tab === 'history') loadHistory();
    if (tab === 'insights') loadInsights();
  });
});

// ── DEMO FILL ──────────────────────────────────────────────────────────────────
fillDemoBtn.addEventListener('click', () => {
  Object.entries(DEMO_VALUES).forEach(([k, v]) => {
    const el = document.getElementById(k);
    if (el) el.value = v;
  });
});

const showWeatherMessage = (text, type = 'error') => {
  if (!weatherMessage) {
    alert(text);
    return;
  }
  weatherMessage.textContent = text;
  weatherMessage.className = `message ${type}`;
  weatherMessage.classList.remove('hidden');
};

const clearWeatherMessage = () => {
  if (!weatherMessage) return;
  weatherMessage.textContent = '';
  weatherMessage.className = 'message hidden';
};

fetchWeatherBtn?.addEventListener('click', async () => {
  const location = locationInput?.value?.trim();
  if (!location) {
    showWeatherMessage('Enter a weather location first.', 'error');
    return;
  }

  const originalText = fetchWeatherBtn.innerHTML;
  fetchWeatherBtn.disabled = true;
  fetchWeatherBtn.innerHTML = 'Fetching...';
  clearWeatherMessage();

  try {
    const res = await fetch('/weather', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ location })
    });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.error || 'Unable to fetch weather');

    document.getElementById('temperature').value = payload.temperature;
    document.getElementById('humidity').value = payload.humidity;
    document.getElementById('rainfall').value = payload.rainfall;
    showWeatherMessage(`Weather values loaded for ${payload.location_name || location}.`, 'success');
  } catch (err) {
    showWeatherMessage('Weather fetch failed: ' + err.message, 'error');
    console.error(err);
  } finally {
    fetchWeatherBtn.disabled = false;
    fetchWeatherBtn.innerHTML = originalText;
  }
});

// ── PREDICTION ──────────────────────────────────────────────────────────────────
form.addEventListener('submit', async (e) => {
  e.preventDefault();

  // UI: loading
  stateIdle.classList.add('hidden');
  stateResult.classList.add('hidden');
  stateLoading.classList.remove('hidden');
  altCard.classList.add('hidden');
  chartCard.classList.add('hidden');
  submitBtn.disabled = true;

  const data = Object.fromEntries(new FormData(form).entries());

  try {
    await new Promise(r => setTimeout(r, 600)); // smooth UX delay

    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    const result = await res.json();

    stateLoading.classList.add('hidden');

    if (!res.ok) {
      alert('Error: ' + result.error);
      stateIdle.classList.remove('hidden');
      return;
    }

    // ── Populate main result ──
    resultEmoji.textContent   = result.crop_info.emoji || '🌱';
    resultCrop.textContent    = result.prediction;
    confidencePct.textContent = result.confidence + '%';
    cropTip.textContent       = result.crop_info.tip || '';
    metaSeason.textContent    = result.crop_info.season || 'N/A';
    metaWater.textContent     = (result.crop_info.water || 'N/A') + ' water needs';

    // Animate confidence bar (after a tick so transition fires)
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        confidenceFill.style.width = result.confidence + '%';
      });
    });

    // Colour bar by confidence
    if (result.confidence >= 80) confidenceFill.style.background = 'linear-gradient(90deg,#16a34a,#22c55e)';
    else if (result.confidence >= 50) confidenceFill.style.background = 'linear-gradient(90deg,#d97706,#f59e0b)';
    else confidenceFill.style.background = 'linear-gradient(90deg,#ef4444,#f87171)';

    stateResult.classList.remove('hidden');

    // ── Top 5 ──
    top5List.innerHTML = '';
    result.top5.forEach((item, i) => {
      const div = document.createElement('div');
      div.className = 'top5-item';
      div.innerHTML = `
        <span class="top5-rank">#${i+1}</span>
        <span class="top5-name">${item.crop}</span>
        <div class="top5-track"><div class="top5-fill ${i===0?'rank1':''}" style="width:0%" data-w="${item.probability}"></div></div>
        <span class="top5-pct">${item.probability}%</span>`;
      top5List.appendChild(div);
    });
    altCard.classList.remove('hidden');
    // Animate bars
    requestAnimationFrame(() => requestAnimationFrame(() => {
      document.querySelectorAll('.top5-fill').forEach(el => el.style.width = el.dataset.w + '%');
    }));

    // ── Feature importance chart ──
    renderImportanceChart(result.feature_importances);
    chartCard.classList.remove('hidden');

  } catch (err) {
    stateLoading.classList.add('hidden');
    stateIdle.classList.remove('hidden');
    alert('Connection error. Make sure the Flask server is running.');
    console.error(err);
  } finally {
    submitBtn.disabled = false;
  }
});

// ── IMPORTANCE CHART ──────────────────────────────────────────────────────────
function renderImportanceChart(importances) {
  const labels = Object.keys(importances).map(k => ({
    N:'Nitrogen', P:'Phosphorus', K:'Potassium',
    temperature:'Temperature', humidity:'Humidity',
    ph:'Soil pH', rainfall:'Rainfall'
  }[k] || k));
  const values = Object.values(importances).map(v => +(v * 100).toFixed(1));

  if (importanceChart) importanceChart.destroy();
  const ctx = document.getElementById('importance-chart').getContext('2d');
  importanceChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: values.map(v => v === Math.max(...values)
          ? 'rgba(22,163,74,0.85)' : 'rgba(22,163,74,0.3)'),
        borderRadius: 8,
        borderSkipped: false,
      }]
    },
    options: {
      responsive: true, indexAxis: 'y',
      plugins: { legend: { display: false }, tooltip: {
        callbacks: { label: ctx => ' ' + ctx.raw + '% influence' }
      }},
      scales: {
        x: { grid: { display: false }, ticks: { callback: v => v + '%', font:{size:11} } },
        y: { grid: { display: false }, ticks: { font:{size:12,weight:'600'} } }
      }
    }
  });
}

// ── HISTORY ──────────────────────────────────────────────────────────────────
async function loadHistory() {
  const list = document.getElementById('history-list');
  try {
    const res = await fetch('/history');
    const data = await res.json();
    if (!data.length) {
      list.innerHTML = `<div class="empty-state"><i class="ph ph-clock-counter-clockwise"></i><p>No predictions yet. Make your first one!</p></div>`;
      return;
    }
    list.innerHTML = data.map(item => {
      const inp = item.inputs;
      return `
        <div class="history-item">
          <div class="history-emoji">${item.emoji}</div>
          <div class="history-info">
            <div class="history-crop">${item.crop}</div>
            <div class="history-params">N:${inp.N} · P:${inp.P} · K:${inp.K} · Temp:${inp.temperature}°C · pH:${inp.ph} · Rain:${inp.rainfall}mm</div>
          </div>
          <div class="history-conf"><i class="ph ph-check-circle"></i>${item.confidence}%</div>
          <div class="history-time">${item.time}</div>
        </div>`;
    }).join('');
  } catch { list.innerHTML = '<p style="color:var(--muted)">Could not load history.</p>'; }
}

// ── INSIGHTS / CROP CHART ─────────────────────────────────────────────────────
async function loadInsights() {
  try {
    const res = await fetch('/dataset-stats');
    const data = await res.json();
    if (!data.crop_counts) return;

    const sorted = Object.entries(data.crop_counts).sort((a,b) => b[1]-a[1]);
    const labels = sorted.map(([k]) => k.charAt(0).toUpperCase() + k.slice(1));
    const values = sorted.map(([,v]) => v);

    if (cropChart) cropChart.destroy();
    const ctx = document.getElementById('crop-chart').getContext('2d');
    cropChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          data: values,
          backgroundColor: 'rgba(22,163,74,0.6)',
          hoverBackgroundColor: 'rgba(22,163,74,0.9)',
          borderRadius: 6,
          borderSkipped: false,
        }]
      },
      options: {
        responsive:true, indexAxis:'y',
        plugins:{ legend:{display:false} },
        scales:{
          x:{ grid:{color:'rgba(0,0,0,0.05)'}, ticks:{font:{size:11}} },
          y:{ grid:{display:false}, ticks:{font:{size:11,weight:'600'}} }
        }
      }
    });
  } catch(e){ console.error(e); }
}


// ── CHATBOT CONTROLLER ───────────────────────────────────────────────────────
(function() {
  // Elements
  const toggleBtn = document.getElementById('chatbot-toggle');
  const container = document.getElementById('chatbot-container');
  const closeBtn = document.getElementById('chat-close');
  const clearBtn = document.getElementById('chat-clear');
  const themeToggle = document.getElementById('chat-theme-toggle');
  const chatBody = document.getElementById('chat-body');
  const chatInput = document.getElementById('chat-input');
  const sendBtn = document.getElementById('chat-send');
  const voiceBtn = document.getElementById('chat-voice');
  const suggestionChips = document.querySelectorAll('.suggestion-chip');

  // User ID Generation / Loading
  let userId = localStorage.getItem('agri_chat_user_id');
  if (!userId) {
    userId = 'farmer_' + Math.random().toString(36).substring(2, 11) + '_' + Date.now();
    localStorage.setItem('agri_chat_user_id', userId);
  }

  // Load Saved Chat Theme
  const savedTheme = localStorage.getItem('agri_chat_theme') || 'light';
  if (savedTheme === 'dark') {
    container.classList.add('dark-theme');
    const themeIcon = themeToggle.querySelector('i');
    if (themeIcon) {
      themeIcon.className = 'ph ph-sun';
    }
  }

  let historyLoaded = false;

  // Toggle Chatbot
  toggleBtn.addEventListener('click', () => {
    container.classList.toggle('open');
    if (container.classList.contains('open')) {
      chatInput.focus();
      if (!historyLoaded) {
        loadChatHistory();
      }
    }
  });

  closeBtn.addEventListener('click', () => {
    container.classList.remove('open');
  });

  // Toggle Theme
  themeToggle.addEventListener('click', () => {
    container.classList.toggle('dark-theme');
    const isDark = container.classList.contains('dark-theme');
    localStorage.setItem('agri_chat_theme', isDark ? 'dark' : 'light');
    const themeIcon = themeToggle.querySelector('i');
    if (themeIcon) {
      themeIcon.className = isDark ? 'ph ph-sun' : 'ph ph-moon';
    }
  });

  // Clear Conversation
  clearBtn.addEventListener('click', () => {
    if (confirm('Are you sure you want to clear your conversation history?')) {
      // Create new user id to effectively reset history on UI & DB queries
      userId = 'farmer_' + Math.random().toString(36).substring(2, 11) + '_' + Date.now();
      localStorage.setItem('agri_chat_user_id', userId);
      
      // Reset Chat Body to initial message
      chatBody.innerHTML = `
        <div class="chat-message bot-msg">
          <div class="chat-bubble">
            Welcome to AgriSmart AI Advisor! 🌾 I can help you with crop recommendations, soil health, fertilizer balances, pest management, and weather impacts. Ask me anything!
          </div>
          <span class="chat-time">Just now</span>
        </div>
      `;
      historyLoaded = true;
    }
  });

  // Helper: Scroll to Bottom
  function scrollToBottom() {
    chatBody.scrollTop = chatBody.scrollHeight;
  }

  // Helper: Format AI Response with basic HTML tags
  function formatResponse(text) {
    // Escape HTML to prevent XSS (except we want to allow tags we generate)
    let formatted = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    // Restore some safe HTML formatting from the fallback model if it had HTML
    formatted = formatted
      .replace(/&lt;strong&gt;/g, "<strong>")
      .replace(/&lt;\/strong&gt;/g, "</strong>")
      .replace(/&lt;em&gt;/g, "<em>")
      .replace(/&lt;\/em&gt;/g, "</em>")
      .replace(/&lt;br&gt;/g, "<br>");

    // Convert **bold** markdown to <strong>bold</strong>
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Convert *italic* markdown to <em>italic</em>
    formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');

    // Convert markdown bullets
    const lines = formatted.split('\n');
    let inList = false;
    const processedLines = lines.map(line => {
      const trimmed = line.trim();
      if (trimmed.startsWith('•') || trimmed.startsWith('-') || trimmed.startsWith('*')) {
        // Strip the bullet symbol
        const content = trimmed.replace(/^[•\-*]\s*/, '').trim();
        let listPrefix = '';
        if (!inList) {
          inList = true;
          listPrefix = '<ul>';
        }
        return listPrefix + '<li>' + content + '</li>';
      } else {
        let suffix = '';
        if (inList) {
          inList = false;
          suffix = '</ul>';
        }
        return suffix + (line ? line + '<br>' : '');
      }
    });
    
    let result = processedLines.join('\n');
    if (inList) {
      result += '</ul>';
    }
    
    return result;
  }

  // Helper: Append Message Bubble to UI
  function appendMessage(sender, text, timeString = null) {
    const time = timeString || new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const msgDiv = document.createElement('div');
    msgDiv.className = `chat-message ${sender === 'user' ? 'user-msg' : 'bot-msg'}`;
    
    const bubbleContent = sender === 'user' ? text : formatResponse(text);
    
    msgDiv.innerHTML = `
      <div class="chat-bubble">
        ${bubbleContent}
      </div>
      <span class="chat-time">${time}</span>
    `;
    
    chatBody.appendChild(msgDiv);
    scrollToBottom();
  }

  // Helper: Append Typing Indicator
  let typingBubble = null;
  function showTypingIndicator() {
    if (typingBubble) return;
    
    typingBubble = document.createElement('div');
    typingBubble.className = 'chat-message bot-msg typing-msg';
    typingBubble.innerHTML = `
      <div class="chat-bubble">
        <div class="typing-indicator">
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
        </div>
      </div>
    `;
    chatBody.appendChild(typingBubble);
    scrollToBottom();
  }

  function hideTypingIndicator() {
    if (typingBubble) {
      typingBubble.remove();
      typingBubble = null;
    }
  }

  // Load Chat History from Backend
  async function loadChatHistory() {
    try {
      const res = await fetch(`/chat-history?user_id=${encodeURIComponent(userId)}`);
      if (!res.ok) throw new Error('Failed to load history');
      const history = await res.json();
      
      if (history && history.length > 0) {
        // Clear initial greeting and load history
        chatBody.innerHTML = '';
        history.forEach(item => {
          let timeStr = null;
          if (item.timestamp) {
            try {
              const dateObj = new Date(item.timestamp.replace(/-/g, '/'));
              timeStr = dateObj.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            } catch(e) {}
          }
          appendMessage('user', item.user, timeStr);
          appendMessage('bot', item.bot, timeStr);
        });
      }
      historyLoaded = true;
    } catch (err) {
      console.error('Error loading chat history:', err);
    }
  }

  // Send Message function
  async function sendMessage(messageText) {
    const text = messageText.trim();
    if (!text) return;

    // Append user message
    appendMessage('user', text);
    chatInput.value = '';
    
    // Show typing
    showTypingIndicator();

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          message: text
        })
      });
      const data = await res.json();
      
      hideTypingIndicator();
      
      if (!res.ok) {
        appendMessage('bot', `⚠️ Error: ${data.error || 'Something went wrong.'}`);
      } else {
        appendMessage('bot', data.response);
      }
    } catch (err) {
      hideTypingIndicator();
      appendMessage('bot', '⚠️ Connection error. Please check if your server is running.');
      console.error(err);
    }
  }

  // Event Listeners for Send
  sendBtn.addEventListener('click', () => {
    sendMessage(chatInput.value);
  });

  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      sendMessage(chatInput.value);
    }
  });

  // Suggestion Chips Click
  suggestionChips.forEach(chip => {
    chip.addEventListener('click', () => {
      sendMessage(chip.textContent);
    });
  });

  // Speech Recognition (Voice Input)
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SpeechRecognition) {
    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    let isRecording = false;

    voiceBtn.addEventListener('click', () => {
      if (isRecording) {
        recognition.stop();
      } else {
        recognition.start();
      }
    });

    recognition.onstart = () => {
      isRecording = true;
      voiceBtn.classList.add('recording');
      voiceBtn.innerHTML = '<i class="ph ph-microphone-slash"></i>';
      chatInput.placeholder = 'Listening... Speak now.';
    };

    recognition.onspeechend = () => {
      recognition.stop();
    };

    recognition.onend = () => {
      isRecording = false;
      voiceBtn.classList.remove('recording');
      voiceBtn.innerHTML = '<i class="ph ph-microphone"></i>';
      chatInput.placeholder = 'Ask a farming question...';
    };

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      chatInput.value = transcript;
      setTimeout(() => {
        sendMessage(transcript);
      }, 500);
    };

    recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
      recognition.stop();
    };
  } else {
    voiceBtn.style.display = 'none';
  }
})();