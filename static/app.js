const FIELDS = [
  ["cgpa", "Cumulative GPA (CGPA)", 0, 5, 0.01, 3.2, ""],
  ["semester_gpa", "Last semester GPA", 0, 5, 0.01, 3.0, ""],
  ["attendance", "Class attendance", 0, 100, 1, 78, "%"],
  ["study_hours", "Self-study per week", 0, 40, 0.5, 10, " hrs"],
  ["assignment_rate", "Assignments submitted", 0, 100, 1, 75, "%"],
  ["carryovers", "Carry-over courses", 0, 10, 1, 1, ""],
  ["sleep_hours", "Sleep per night", 3, 10, 0.5, 6.5, " hrs"],
];
const TONE = {Excellent: "var(--green)", Good: "var(--blue)", "At Risk": "var(--amber)", Critical: "var(--coral)"};
const LEVEL = {high: "var(--coral)", medium: "var(--amber)", low: "var(--green)"};
const $ = (s) => document.querySelector(s);

// tabs
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("on", x === t));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("on", v.id === t.dataset.tab));
}));

// sliders
$("#sliders").innerHTML = FIELDS.map(([id, label, min, max, step, val, unit]) => `
  <div class="field"><div class="head"><label for="${id}">${label}</label><output id="o_${id}">${val}${unit}</output></div>
  <input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></div>`).join("");
FIELDS.forEach(([id, , , , , , unit]) => $("#" + id).addEventListener("input", (e) => ($("#o_" + id).textContent = e.target.value + unit)));

// predict
$("#profile").addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = {level: $("#level").value};
  FIELDS.forEach(([id]) => (body[id] = $("#" + id).value));
  const err = $("#error"); err.hidden = true;
  try {
    const res = await fetch("/api/predict", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);
    show(data);
  } catch (ex) { err.textContent = ex.message; err.hidden = false; }
});

function show(d) {
  $("#empty").hidden = true; $("#out").hidden = false;
  const tone = TONE[d.standing];
  $("#out").style.setProperty("--tone", tone);
  $("#standing").textContent = d.standing;
  $("#summary").textContent = d.summary;
  $("#pct").textContent = Math.round(d.confidence * 100) + "%";
  $("#arc").style.strokeDashoffset = 415 * (1 - d.confidence);
  $("#bars").innerHTML = Object.entries(d.probabilities).reverse().map(([k, v]) =>
    `<div class="bar"><span style="text-align:left">${k}</span><i><b style="background:${TONE[k]}" data-w="${v * 100}"></b></i><span>${Math.round(v * 100)}%</span></div>`).join("");
  requestAnimationFrame(() => document.querySelectorAll(".bar b").forEach((b) => (b.style.width = b.dataset.w + "%")));
  $("#advice").innerHTML = d.advice.map((a) =>
    `<li style="--c:${LEVEL[a.level]}"><strong>${a.title}</strong><span>${a.detail}</span></li>`).join("");
}

// GPA calculator
function addCourse(code = "", units = 3, grade = "A") {
  const row = document.createElement("div"); row.className = "crs";
  row.innerHTML = `<input placeholder="Course code" value="${code}" aria-label="Course code">
    <input type="number" min="1" max="10" value="${units}" aria-label="Units">
    <select aria-label="Grade">${"ABCDEF".split("").map((g) => `<option ${g === grade ? "selected" : ""}>${g}</option>`).join("")}</select>
    <button type="button" aria-label="Remove course">&times;</button>`;
  row.querySelector("button").onclick = () => row.remove();
  $("#courses").appendChild(row);
}
["CSC 303", "CSC 305", "MTH 301", "GST 311"].forEach((c, i) => addCourse(c, [3, 3, 3, 2][i], "AABA"[i]));
$("#addCourse").onclick = () => addCourse();
$("#calc").onclick = async () => {
  const courses = [...document.querySelectorAll(".crs")].map((r) => {
    const [c, u, g] = r.querySelectorAll("input,select");
    return {code: c.value, units: u.value, grade: g.value};
  });
  const res = await fetch("/api/gpa", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({courses})});
  const d = await res.json();
  $("#gpaOut").hidden = false;
  $("#gpaVal").textContent = res.ok ? d.gpa.toFixed(2) : "!";
  $("#gpaClass").textContent = res.ok ? `${d.class}, ${d.units} units` : d.error;
};
