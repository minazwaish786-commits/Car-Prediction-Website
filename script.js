'use strict';

// Categorical encodings copied from the supplied notebook.
const fuelMap = { Petrol: 0, Diesel: 1, CNG: 2 };
const sellerMap = { Dealer: 0, Individual: 1 };
const transmissionMap = { Manual: 0, Automatic: 1 };
let model = null;

const $ = id => document.getElementById(id);
const statusBox = $('datasetStatus');
const errorBox = $('error');
const resultBox = $('result');

function parseCSV(text) {
  const rows = [];
  let row = [], value = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === '"' && quoted && text[i + 1] === '"') { value += '"'; i++; }
    else if (ch === '"') quoted = !quoted;
    else if (ch === ',' && !quoted) { row.push(value.trim()); value = ''; }
    else if ((ch === '\n' || ch === '\r') && !quoted) {
      row.push(value.trim());
      if (row.some(v => v !== '')) rows.push(row);
      row = []; value = '';
      if (ch === '\r' && text[i + 1] === '\n') i++;
    } else value += ch;
  }
  row.push(value.trim());
  if (row.some(v => v !== '')) rows.push(row);
  if (quoted) throw new Error('CSV file mein quotation marks properly close nahi hue.');
  if (!rows.length) throw new Error('CSV file empty hai.');
  const headers = rows.shift().map(h => h.replace(/^\uFEFF/, ''));
  return rows.map(values => Object.fromEntries(headers.map((h, i) => [h, values[i] ?? ''])));
}

function encodeRow(row) {
  return [Number(row.Year), Number(row.Present_Price), Number(row.Kms_Driven),
    fuelMap[row.Fuel_Type], sellerMap[row.Seller_Type],
    transmissionMap[row.Transmission], Number(row.Owner)];
}

// Solve the normal equations using Gaussian elimination and partial pivoting.
function solveLinearSystem(matrix, vector) {
  const n = vector.length;
  const a = matrix.map((r, i) => [...r, vector[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++)
      if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
    if (Math.abs(a[pivot][col]) < 1e-12)
      throw new Error('Regression model could not be fitted. Please check the CSV data.');
    [a[col], a[pivot]] = [a[pivot], a[col]];
    const divisor = a[col][col];
    for (let j = col; j <= n; j++) a[col][j] /= divisor;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = a[r][col];
      for (let j = col; j <= n; j++) a[r][j] -= factor * a[col][j];
    }
  }
  return a.map(r => r[n]);
}

function trainModel(rows) {
  const required = ['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner'];
  if (!rows.length || !required.every(k => Object.prototype.hasOwnProperty.call(rows[0], k)))
    throw new Error('CSV columns expected format se match nahi karte. Original car_data_2.csv upload karein.');
  const valid = rows.filter(row => {
    const x = encodeRow(row);
    return x.every(Number.isFinite) && Number.isFinite(Number(row.Selling_Price));
  });
  if (valid.length < 10) throw new Error('Dataset mein kam se kam 10 valid records hone chahiye.');

  const X = valid.map(encodeRow), y = valid.map(r => Number(r.Selling_Price));
  const means = Array(7).fill(0), scales = Array(7).fill(0);
  X.forEach(x => x.forEach((v, j) => means[j] += v / X.length));
  X.forEach(x => x.forEach((v, j) => scales[j] += (v - means[j]) ** 2));
  scales.forEach((v, j) => scales[j] = Math.sqrt(v / X.length) || 1);

  const Z = X.map(x => [1, ...x.map((v, j) => (v - means[j]) / scales[j])]);
  const n = 8, A = Array.from({length:n}, () => Array(n).fill(0)), B = Array(n).fill(0);
  for (let i = 0; i < Z.length; i++) {
    for (let j = 0; j < n; j++) {
      B[j] += Z[i][j] * y[i];
      for (let k = 0; k < n; k++) A[j][k] += Z[i][j] * Z[i][k];
    }
  }
  // Very small ridge term improves numerical stability; this remains an OLS-style estimate.
  for (let j = 1; j < n; j++) A[j][j] += 1e-8;
  model = {means, scales, coefficients: solveLinearSystem(A, B)};
  $('recordCount').textContent = valid.length;
  statusBox.classList.add('ready');
  statusBox.innerHTML = `<span class="status-dot"></span><div><b>Dataset loaded and model trained</b><small>${valid.length} valid records are ready for prediction.</small></div>`;
  $('predictBtn').disabled = false;
  $('predictBtn').innerHTML = 'Predict selling price <span>→</span>';
}

function predict(features) {
  const z = [1, ...features.map((v, j) => (v - model.means[j]) / model.scales[j])];
  return z.reduce((sum, v, i) => sum + v * model.coefficients[i], 0);
}

$('csvFile').addEventListener('change', event => {
  const file = event.target.files[0];
  if (!file) return;
  model = null; $('predictBtn').disabled = true; resultBox.classList.add('hidden'); errorBox.textContent = '';
  const reader = new FileReader();
  reader.onload = () => {
    try { trainModel(parseCSV(String(reader.result))); }
    catch (err) {
      statusBox.classList.remove('ready');
      statusBox.innerHTML = '<span class="status-dot"></span><div><b>Dataset could not be loaded</b><small>Please check the selected CSV file.</small></div>';
      errorBox.textContent = err.message;
    }
  };
  reader.onerror = () => { errorBox.textContent = 'CSV file read nahi ho payi.'; };
  reader.readAsText(file);
});

$('carForm').addEventListener('submit', event => {
  event.preventDefault(); errorBox.textContent = ''; resultBox.classList.add('hidden');
  if (!model) { errorBox.textContent = 'Prediction se pehle car_data_2.csv upload karein.'; return; }
  const year = Number($('year').value), price = Number($('presentPrice').value);
  const kms = Number($('kmsDriven').value), owner = Number($('owner').value);
  if (!Number.isFinite(year) || year < 1900 || year > 2100 || !Number.isFinite(price) || price <= 0 || !Number.isFinite(kms) || kms < 0 || ![0,1,2,3].includes(owner)) {
    errorBox.textContent = 'Please valid car details enter karein.'; return;
  }
  const features = [year, price, kms, fuelMap[$('fuel').value], sellerMap[$('seller').value], transmissionMap[$('transmission').value], owner];
  const estimate = predict(features);
  if (!Number.isFinite(estimate)) { errorBox.textContent = 'Prediction calculate nahi ho payi. Input details check karein.'; return; }
  $('price').innerHTML = `₹ ${Math.max(0, estimate).toFixed(2)} <small>Lakhs</small>`;
  resultBox.classList.remove('hidden');
  resultBox.scrollIntoView({behavior:'smooth', block:'center'});
});

$('resetBtn').addEventListener('click', () => {
  resultBox.classList.add('hidden');
  $('carForm').scrollIntoView({behavior:'smooth', block:'center'});
});
