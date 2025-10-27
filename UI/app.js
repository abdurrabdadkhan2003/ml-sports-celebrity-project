const celebImages = {
  "Lionel Messi": "images/messi.jpeg",
  "Maria Sharapova": "images/sharapova.jpeg",
  "Roger Federer": "images/federer.jpeg",
  "Serena Williams": "images/serena.jpeg",
  "Virat Kohli": "images/virat.jpeg"
};

Dropzone.autoDiscover = false;
const dz = new Dropzone('#dropzone', {
  url: "#",
  maxFiles: 1,
  addRemoveLinks: true,
  dictDefaultMessage: '<span style="color:#39b144;">Drop or click to upload a celebrity photo</span>',
  clickable: true,
  autoProcessQueue: false,
  previewTemplate: `<div class="dz-preview dz-file-preview">
    <img data-dz-thumbnail class="dz-image-preview rounded" style="width:100px;height:100px;border-radius:15px;margin-top:10px;box-shadow:0 0 12px #50f7da77;"/>
    <div class="dz-details" style="margin-top:0.4rem;">
      <div class="dz-filename"><span data-dz-name></span></div>
      <a href="javascript:;" data-dz-remove class="dz-remove" style="font-weight:700;color:#24d680;">Remove file</a>
    </div>
  </div>`
});
dz.on("addedfile", function(file) {
  document.getElementById("classifyBtn").style.display = "block";
});
dz.on("removedfile", function() {
  document.getElementById("classifyBtn").style.display = "none";
  document.getElementById("results").innerHTML = "";
});
document.getElementById('classifyBtn').onclick = function() {
  if (dz.files.length === 0) {
    alert("Please select or upload an image first!");
    return;
  }
  document.getElementById("results").innerHTML = `<div style="color:#24d680;"><span class="loader"></span> Classifying...</div>`;
  setTimeout(() => {
    // DEMO: Dummy results only!
    showResults([
      {name:"Lionel Messi", probability:4.15},
      {name:"Maria Sharapova", probability:2.62},
      {name:"Roger Federer", probability:63.88},
      {name:"Serena Williams", probability:23.66},
      {name:"Virat Kohli", probability:5.69}
    ]);
    throwConfetti();
  }, 1300);
};

function showResults(list){
  let html = `<table class="results-table"><tr>
    <th>Profile</th>
    <th>Player</th>
    <th>Probability Score</th>
    </tr>`;
  list.forEach((row, idx) => {
    html += `<tr class="${idx === 2 ? "result-winner" : ""}">
      <td class="profile-cell"><img src="${celebImages[row.name]||'images/unknown.jpg'}" alt="${row.name}"></td>
      <td>${row.name}</td>
      <td>${row.probability.toFixed(2)}</td>
    </tr>`;
  });
  html += `</table>`;
  document.getElementById("results").innerHTML = html;
}
// Confetti Animation
function throwConfetti() {
  for (let i = 0; i < 90; i++) {
    let confetti = document.createElement('div');
    confetti.className = 'confetti';
    confetti.style.left = Math.random() * 100 + 'vw';
    confetti.style.animationDelay = (Math.random() * 2) + 's';
    confetti.style.background = `linear-gradient(110deg,#39b144,#ffd600,#39f6f1, #3a6fa0)`;
    document.body.appendChild(confetti);
    setTimeout(() => confetti.remove(), 3200);
  }
}
// Loader animation
const style = document.createElement('style');
style.innerHTML = `
.loader,
.loader:before,
.loader:after {
  background: #24d680; border-radius: 50%; width: 0.85em; height: 0.85em; animation: load1 1.2s infinite ease-in-out;
}
.loader { color: #24d680; font-size: 11px; margin: 0 6px; position: relative; text-indent: -9999em; transform: translateZ(0); animation-delay: -0.16s;}
.loader:before, .loader:after { content: ''; position: absolute; top: 0;}
.loader:before { left: -1.1em; animation-delay: -0.32s;}
.loader:after { left: 1.1em;}
@keyframes load1 {
  0%, 80%, 100% { box-shadow: 0 2.4em 0 -1.3em;}
  40% { box-shadow: 0 2.4em 0 0;}
}
`; document.head.appendChild(style);
