/* Constants */
const CELL = 16, N = 16, NUM_ACTIONS = 4;
const DIRS = [{dx:0,dy:-1},{dx:0,dy:1},{dx:-1,dy:0},{dx:1,dy:0}];
const MAX_MOVES = 250;
const POUNCE_COOLDOWN = 4, POUNCE_DIST = 3;
const GOAP_SPRINT = 5, GOAP_REST = 2;
const MAX_ENERGY = 80, ENERGY_MOVE = 1, ENERGY_POUNCE = 5;
const GOAP_START = {x:1, y:1};
const RL_START   = {x:N-2, y:N-2};
const ZONE_MIN = 3, ZONE_MAX = 6;

/* Canvas refs */
const canvas       = document.getElementById('cv');
const ctx          = canvas.getContext('2d');
const chartCanvas  = document.getElementById('chart');
const cctx         = chartCanvas.getContext('2d');

/* State */
let grid = [], foodCell = {x:8, y:8};
let staticGrid = null, staticFood = null;
let qtable = {}, rewardHistory = [];
let trained = false, running = false, stopFlag = false;
let modules = {pounce:true, rest:true, energy:true, randomenv:false};
let windowHalf = 4;

/* Colors */
const C = {
  free:     '#e0e0e0',
  wall:     '#0a0a0a',
  goap:     '#2563eb',
  goapRest: '#1d9e75',
  goapDead: '#555',
  rl:       '#a78bfa',
  path:     '#eb7f04',
  pounce:   '#dc2626',
  food:     '#26dc44',
};

/* UI helpers */
function updateWindowLabel(v) {
  windowHalf = Math.floor(v / 2);
  document.getElementById('window-val').textContent = v + '×' + v;
}

function onRunsSlider(v) {
  document.getElementById('runs-val').textContent = v;
  document.getElementById('trainBtn').textContent = `Train RL (${v} Runs)`;
}

function getTrainEpisodes() {
  return parseInt(document.getElementById('runs-slider').value);
}

function toggleMod(name) {
  modules[name] = !modules[name];
  const on = modules[name];
  document.getElementById(name + '-track').classList.toggle('on', on);

  if (name === 'randomenv') {
    document.getElementById('randomenv-badge').className = on ? 'badge badge-on' : 'badge badge-off';
    document.getElementById('randomenv-badge').textContent = on ? 'ON' : 'OFF';
    document.getElementById('randenv-desc').textContent = on
      ? 'New Map each Episode'
      : 'Fixed Map for all Episodes';
  } else {
    document.getElementById(name + '-badge').className = on ? 'badge badge-on' : 'badge badge-off';
    document.getElementById(name + '-badge').textContent = on ? 'ON' : 'OFF';
  }

  if (name === 'rest')   document.getElementById('stamina-section').style.opacity = on ? '1' : '0.3';
  if (name === 'energy') document.getElementById('energy-section').style.opacity  = on ? '1' : '0.3';
}

/* Grid construction */
function buildGrid() {
  const density = parseInt(document.getElementById('density-slider').value) / 100;
  const g = [];
  for (let y = 0; y < N; y++) {
    const row = [];
    for (let x = 0; x < N; x++)
      row.push((x===0||y===0||x===N-1||y===N-1) ? 0 : Math.random() < density ? 1 : 0);
    g.push(row);
  }
  // Clear spawn areas
  g[GOAP_START.y][GOAP_START.x] = 0;
  g[GOAP_START.y+1][GOAP_START.x] = 0;
  g[GOAP_START.y][GOAP_START.x+1] = 0;
  g[RL_START.y][RL_START.x] = 0;
  g[RL_START.y-1][RL_START.x] = 0;
  g[RL_START.y][RL_START.x-1] = 0;
  const fc = {x: 7 + Math.floor(Math.random()*3), y: 7 + Math.floor(Math.random()*3)};
  g[fc.y][fc.x] = 0;
  return {g, fc};
}

function makeGrid() {
  const {g, fc} = buildGrid();
  grid = g; foodCell = fc;
  staticGrid = g.map(r => [...r]);
  staticFood = {...fc};
}

function loadEpisodeMap() {
  if (modules.randomenv) {
    const {g, fc} = buildGrid();
    grid = g; foodCell = fc;
  } else {
    grid = staticGrid.map(r => [...r]);
    foodCell = {...staticFood};
  }
  return {
    gridSnapshot: grid.map(r => [...r]),
    foodSnapshot: {...foodCell},
  };
}

/* Utility */
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function passable(x, y)    { return x>=0 && y>=0 && x<N && y<N && grid[y][x]===0; }
function dist(a, b)        { return Math.abs(a.x-b.x) + Math.abs(a.y-b.y); }

function mobilityCount(pos) {
  return DIRS.filter(({dx,dy}) => passable(pos.x+dx, pos.y+dy)).length;
}

/* Reward function */
function computeReward(rlPos, goapPos, caught, survived) {
  if (caught)   return -200;
  if (survived) return  300;

  const d = dist(rlPos, goapPos);
  let r = 0;

  // Zone reward – goldilocks band [ZONE_MIN, ZONE_MAX]
  if (d >= ZONE_MIN && d <= ZONE_MAX) {
    r += 3.0;
  } else if (d < ZONE_MIN) {
    r -= (ZONE_MIN - d) * 4.0;
  } else {
    r += Math.max(0, 2.0 - (d - ZONE_MAX) * 0.3);
  }

  // Mobility penalty – prefer open positions over corners
  const mob = mobilityCount(rlPos);
  if (mob <= 1) r -= 3.0;
  else if (mob === 2) r -= 1.0;

  return r;
}

/* A* pathfinding */
function astar(start, goal) {
  const key = p => `${p.x},${p.y}`;
  const open = [{...start, g:0, h:dist(start,goal), f:dist(start,goal), parent:null}];
  const closed = new Map();

  while (open.length) {
    open.sort((a,b) => a.f - b.f);
    const cur = open.shift();
    if (cur.x===goal.x && cur.y===goal.y) {
      const path = []; let n = cur;
      while (n) { path.unshift({x:n.x, y:n.y}); n = n.parent; }
      return path;
    }
    const ck = key(cur);
    if (closed.has(ck)) continue;
    closed.set(ck, cur);
    for (const {dx,dy} of DIRS) {
      const nx=cur.x+dx, ny=cur.y+dy;
      if (!passable(nx,ny)) continue;
      const nk = `${nx},${ny}`;
      if (closed.has(nk)) continue;
      const g = cur.g+1, h = dist({x:nx,y:ny}, goal);
      const ex = open.find(o => o.x===nx && o.y===ny);
      if (!ex || g < ex.g) {
        const nd = {x:nx, y:ny, g, h, f:g+h, parent:cur};
        if (!ex) open.push(nd); else Object.assign(ex, nd);
      }
    }
  }
  return null;
}

/* Pounce logic */
function getPounceTarget(from, path) {
  if (!modules.pounce || !path || path.length < 2) return null;
  const idx = path.findIndex(p => p.x===from.x && p.y===from.y);
  if (idx < 0 || idx+1 >= path.length) return null;
  const dx = Math.sign(path[idx+1].x - from.x);
  const dy = Math.sign(path[idx+1].y - from.y);
  const targets = [];
  for (let i=1; i<=POUNCE_DIST; i++) {
    const nx=from.x+dx*i, ny=from.y+dy*i;
    if (!passable(nx,ny)) break;
    targets.push({x:nx, y:ny});
  }
  return targets.length === POUNCE_DIST ? targets : null;
}

/* GOAP step */
function goapStep(gs, rlPos) {
  let {pos, pounceCooldown, sprintCount, restCount, energy} = gs;
  const exhausted = modules.energy && energy <= 0;

  if (modules.rest && restCount > 0) {
    return {
      pos, resting:true, exhausted, energy, pounceUsed:false,
      pathCells:[], atFood:false,
      pounceCooldown: modules.pounce ? Math.max(0, pounceCooldown-1) : 0,
      sprintCount, restCount: restCount-1,
    };
  }

  if (exhausted) {
    const foodPath = astar(pos, foodCell);
    if (foodPath && foodPath.length > 1) {
      const np = foodPath[1];
      const ate = np.x===foodCell.x && np.y===foodCell.y;
      const ns  = modules.rest ? sprintCount+1 : 0;
      const nr  = (modules.rest && ns >= GOAP_SPRINT) ? GOAP_REST : 0;
      return {
        pos:np, resting:false, exhausted:true, energy: ate ? MAX_ENERGY : energy,
        pounceUsed:false, pathCells:foodPath, atFood:ate, pounceCooldown:0,
        sprintCount: nr>0 ? 0 : ns, restCount:nr,
      };
    }
    return {pos, resting:false, exhausted:true, energy, pounceUsed:false, pathCells:[], atFood:false, pounceCooldown:0, sprintCount, restCount:0};
  }

  const path = astar(pos, rlPos);
  const ns = modules.rest ? sprintCount+1 : 0;
  const nr = (modules.rest && ns >= GOAP_SPRINT) ? GOAP_REST : 0;
  const nextSprint = nr > 0 ? 0 : ns;

  if (!path || path.length < 2) {
    const e = modules.energy ? Math.max(0, energy-ENERGY_MOVE) : MAX_ENERGY;
    return {
      pos, resting:false, exhausted:false, energy:e, pounceUsed:false,
      pathCells:[], atFood:false, sprintCount:nextSprint, restCount:nr,
      pounceCooldown: modules.pounce ? Math.max(0, pounceCooldown-1) : 0,
    };
  }

  if (modules.pounce && pounceCooldown > 0) {
    const e = modules.energy ? Math.max(0, energy-ENERGY_MOVE) : MAX_ENERGY;
    return {pos:path[1], resting:false, exhausted:false, energy:e, pounceUsed:false, pathCells:path, atFood:false, pounceCooldown:pounceCooldown-1, sprintCount:nextSprint, restCount:nr};
  }

  if (modules.pounce) {
    const pounce = getPounceTarget(pos, path);
    const hitsTarget = pounce && pounce[POUNCE_DIST-1].x===rlPos.x && pounce[POUNCE_DIST-1].y===rlPos.y;
    const pi = pounce ? path.findIndex(p=>p.x===pos.x&&p.y===pos.y) : -1;
    const li = pounce ? path.findIndex(p=>p.x===pounce[POUNCE_DIST-1].x&&p.y===pounce[POUNCE_DIST-1].y) : -1;
    const saving = li >= 0 ? (li-pi-POUNCE_COOLDOWN) : 0;
    const canPounce = !modules.energy || energy >= ENERGY_POUNCE;
    if (canPounce && pounce && (hitsTarget || saving>0)) {
      const e = modules.energy ? Math.max(0, energy-ENERGY_POUNCE) : MAX_ENERGY;
      return {pos:pounce[POUNCE_DIST-1], resting:false, exhausted:false, energy:e, pounceUsed:true, pathCells:path, pounceTrail:pounce, atFood:false, pounceCooldown:POUNCE_COOLDOWN, sprintCount:nextSprint, restCount:nr};
    }
  }

  const np = path[1];
  const ate = modules.energy && np.x===foodCell.x && np.y===foodCell.y;
  const e   = ate ? MAX_ENERGY : modules.energy ? Math.max(0, energy-ENERGY_MOVE) : MAX_ENERGY;
  return {pos:np, resting:false, exhausted:false, energy:e, pounceUsed:false, pathCells:path, atFood:ate, pounceCooldown:0, sprintCount:nextSprint, restCount:nr};
}

/* Q-learning state key */
function stateKey(rlPos, goapPos, pounceCooldown, restCount, energy) {
  const wh  = windowHalf;
  const rdx = clamp(rlPos.x-goapPos.x, -wh, wh) + wh;
  const rdy = clamp(rlPos.y-goapPos.y, -wh, wh) + wh;
  const cd  = modules.pounce ? Math.min(pounceCooldown, 4) : 0;
  const rc  = modules.rest   ? Math.min(restCount, GOAP_REST) : 0;
  const eb  = modules.energy ? Math.min(4, Math.floor((MAX_ENERGY-energy)/20)) : 0;
  const mob = Math.min(mobilityCount(rlPos), 4);
  return `${rdx}_${rdy}_${cd}_${rc}_${eb}_${mob}`;
}

function getQ(sk, a) { return (qtable[sk] || (qtable[sk] = [0,0,0,0]))[a]; }
function setQ(sk, a, v) { (qtable[sk] || (qtable[sk] = [0,0,0,0]))[a] = v; }

function validActions(pos) {
  return DIRS.map((_,i) => i).filter(i => { const {dx,dy}=DIRS[i]; return passable(pos.x+dx, pos.y+dy); });
}

function chooseAction(sk, epsilon, pos) {
  const valid = validActions(pos);
  if (!valid.length) return 0;
  if (Math.random() < epsilon) return valid[Math.floor(Math.random()*valid.length)];
  const qs = qtable[sk] || [0,0,0,0];
  let best = valid[0];
  for (const a of valid) if (qs[a] > qs[best]) best = a;
  return best;
}

function rlMove(pos, action) {
  const {dx,dy} = DIRS[action];
  const nx=pos.x+dx, ny=pos.y+dy;
  return passable(nx,ny) ? {x:nx,y:ny} : pos;
}

/* Episode runner */
function runEpisode(epsilon, render) {
  const {gridSnapshot, foodSnapshot} = loadEpisodeMap();
  let gs     = {pos:{...GOAP_START}, pounceCooldown:0, sprintCount:0, restCount:0, energy:MAX_ENERGY};
  let rlPos  = {...RL_START};
  let totalReward=0, moves=0, caught=false;
  const lr=0.15, gamma=0.95;
  const frames = [];

  while (moves < MAX_MOVES && !caught) {
    const sk     = stateKey(rlPos, gs.pos, gs.pounceCooldown, gs.restCount, gs.energy);
    const action = chooseAction(sk, epsilon, rlPos);
    const newRL  = rlMove(rlPos, action);
    const ng     = goapStep(gs, newRL);
    caught       = ng.pos.x===newRL.x && ng.pos.y===newRL.y;
    const survived = !caught && moves===MAX_MOVES-1;
    const reward = computeReward(newRL, ng.pos, caught, survived);
    totalReward += reward;

    const nsk  = stateKey(newRL, ng.pos, ng.pounceCooldown, ng.restCount, ng.energy);
    const mxQ  = Math.max(...(qtable[nsk] || [0,0,0,0]));
    const oldQ = getQ(sk, action);
    setQ(sk, action, oldQ + lr*(reward + gamma*mxQ - oldQ));

    if (render) {
      frames.push({
        rl:           {...newRL},
        goap:         {...ng.pos},
        path:         ng.pathCells,
        pounceTrail:  ng.pounceTrail,
        pounceCooldown: ng.pounceCooldown,
        sprintCount:  ng.sprintCount,
        restCount:    ng.restCount,
        resting:      ng.resting,
        exhausted:    ng.exhausted,
        energy:       ng.energy,
        atFood:       ng.atFood,
        caught, moves,
        gridSnapshot, foodSnapshot,
      });
    }
    rlPos = newRL;
    gs    = {pos:ng.pos, pounceCooldown:ng.pounceCooldown, sprintCount:ng.sprintCount, restCount:ng.restCount, energy:ng.energy};
    moves++;
    if (caught) break;
  }
  return {totalReward, survived: !caught && moves>=MAX_MOVES, moves, frames};
}

/* Drawing */
function drawZoneRing(goapPos) {
  octx.clearRect(0, 0, 256, 256);
  if (!goapPos) return;
  const cx = (goapPos.x + 0.5) * CELL;
  const cy = (goapPos.y + 0.5) * CELL;
  octx.strokeStyle = 'rgba(255,255,255,0.08)';
  octx.lineWidth   = 1;
  octx.setLineDash([3, 3]);
  [ZONE_MIN, ZONE_MAX].forEach(r => {
    octx.beginPath();
    octx.arc(cx, cy, r * CELL, 0, Math.PI*2);
    octx.stroke();
  });
  octx.setLineDash([]);
}

function drawState(rlPos, goapPos, pathCells, pounceTrail, resting, exhausted, gridSnap, foodSnap) {
  const g    = gridSnap || grid;
  const food = foodSnap  || foodCell;
  ctx.clearRect(0, 0, 256, 256);

  const pathSet   = new Set((pathCells   || []).map(p => `${p.x},${p.y}`));
  const pounceSet = new Set((pounceTrail || []).map(p => `${p.x},${p.y}`));

  for (let y=0; y<N; y++) for (let x=0; x<N; x++) {
    const k = `${x},${y}`;
    let c = g[y][x] === 1 ? C.wall : C.free;
    if (pathSet.has(k))   c = C.path;
    if (pounceSet.has(k)) c = C.pounce;
    if (modules.energy && x===food.x && y===food.y) c = C.food;
    ctx.fillStyle = c;
    ctx.fillRect(x*CELL, y*CELL, CELL, CELL);
    ctx.strokeStyle='rgba(0,0,0,0.06)';
    ctx.strokeRect(x*CELL,y*CELL,CELL,CELL);
  }

  if (rlPos) {
    ctx.fillStyle = C.rl;
    ctx.fillRect(rlPos.x*CELL, rlPos.y*CELL, CELL, CELL);
  }
  if (goapPos) {
    ctx.fillStyle = exhausted ? C.goapDead : resting ? C.goapRest : C.goap;
    ctx.fillRect(goapPos.x*CELL, goapPos.y*CELL, CELL, CELL);
    if (resting || exhausted) {
      ctx.fillStyle = 'rgba(255,255,255,0.35)';
      ctx.fillRect(goapPos.x*CELL+4, goapPos.y*CELL+4, 8, 8);
    }
  }
}

/* Energy bar */
function updateEnergyBar(energy) {
  const pct  = Math.round(energy / MAX_ENERGY * 100);
  const fill = document.getElementById('energy-fill');
  fill.style.width      = pct + '%';
  fill.style.background = pct>50 ? '#08d419' : pct>20 ? '#f59611' : '#f31717';
  document.getElementById('energy-label').textContent = energy + '/' + MAX_ENERGY;
}

/* Stamina bar */
function updateStaminaBar(sprintCount, restCount) {
  for (let i=0; i<GOAP_SPRINT; i++) {
    const el = document.getElementById('sc'+i);
    el.classList.remove('active', 'filled');
    if      (restCount > 0)   el.classList.add('filled');
    else if (i < sprintCount) el.classList.add('filled');
    else if (i===sprintCount) el.classList.add('active');
  }
  document.getElementById('rest-label').textContent =
    restCount > 0 ? `Resting (${restCount} left)` : `Sprint ${sprintCount}/${GOAP_SPRINT}`;
}

/* Reward chart */
function drawChart() {
  const w=chartCanvas.width, h=chartCanvas.height;
  cctx.clearRect(0,0,w,h);
  if (rewardHistory.length < 2) return;
  const mn=Math.min(...rewardHistory), mx=Math.max(...rewardHistory), range=mx-mn||1;
  const smooth=[], win=20;
  for (let i=0; i<rewardHistory.length; i++) {
    const s=Math.max(0,i-win), sl=rewardHistory.slice(s,i+1);
    smooth.push(sl.reduce((a,b)=>a+b,0)/sl.length);
  }
  cctx.strokeStyle='rgba(99,102,241,0.25)'; cctx.lineWidth=1;
  cctx.beginPath();
  rewardHistory.forEach((v,i)=>{const x=i/(rewardHistory.length-1)*w,y=h-(v-mn)/range*h;i===0?cctx.moveTo(x,y):cctx.lineTo(x,y);});
  cctx.stroke();
  cctx.strokeStyle='#6366f1'; cctx.lineWidth=2;
  cctx.beginPath();
  smooth.forEach((v,i)=>{const x=i/(smooth.length-1)*w,y=h-(v-mn)/range*h;i===0?cctx.moveTo(x,y):cctx.lineTo(x,y);});
  cctx.stroke();
}

/* Log */
function log(html) {
  const d = document.getElementById('log');
  const l = document.createElement('div');
  l.className   = 'log-line';
  l.innerHTML   = html;
  d.appendChild(l);
  d.scrollTop   = d.scrollHeight;
}
function clearLog() { document.getElementById('log').innerHTML = ''; }
function delay(ms)  { return new Promise(r => setTimeout(r, ms)); }

/* Training */
async function startTraining() {
  if (running) return;
  const TRAIN_EPISODES = getTrainEpisodes();
  running=true; stopFlag=false;
  document.getElementById('trainBtn').disabled = true;
  document.getElementById('playBtn').disabled  = true;
  clearLog(); rewardHistory=[]; qtable={}; trained=false;

  const wsize   = windowHalf*2+1;
  const mods    = Object.entries(modules).filter(([k,v])=>v&&k!=='randomenv').map(([k])=>k).join('+');
  const envMode = modules.randomenv ? 'Randomized' : 'Fixed';

  log(`<span class="kw"># Q-Learning — ${mods||'base'} — ${envMode} Env — ${TRAIN_EPISODES} Episodes</span>`);
  log(`  window ${wsize}×${wsize}  zone [${ZONE_MIN}–${ZONE_MAX}]  mobility penalty ON`);

  const {gridSnapshot:gs0, foodSnapshot:fs0} = loadEpisodeMap();
  drawState(RL_START, GOAP_START, [], [], false, false, gs0, fs0);
  await delay(40);

  let survived=0, caught=0;
  const BATCH=10;
  for (let ep=0; ep<TRAIN_EPISODES && !stopFlag; ep+=BATCH) {
    const end = Math.min(ep+BATCH, TRAIN_EPISODES);
    for (let e=ep; e<end; e++) {
      const eps = Math.max(0.05, 1.0-e/TRAIN_EPISODES*1.4);
      const r   = runEpisode(eps, false);
      rewardHistory.push(r.totalReward);
      if (r.survived) survived++; else caught++;
    }
    const pct = Math.round(end/TRAIN_EPISODES*100);
    document.getElementById('prog').style.width = pct+'%';
    document.getElementById('prog-label').textContent = `episode ${end}/${TRAIN_EPISODES}`;
    drawChart();
    if (ep % Math.max(50, Math.round(TRAIN_EPISODES/10)) === 0)
      log(`  EP ${end}: R=${Math.round(rewardHistory[rewardHistory.length-1])}, Survived=${survived}, Caught=${caught}`);
    await delay(0);
  }

  trained=true; running=false;
  document.getElementById('trainBtn').disabled = false;
  document.getElementById('playBtn').disabled  = false;
  document.getElementById('prog-label').textContent = 'training complete';
  log(`<br><span class="ok"># Done — ${survived}/${TRAIN_EPISODES} Survived (${Math.round(survived/TRAIN_EPISODES*100)}%)</span>`);
  log(`  Q-Table: ${Object.keys(qtable).length} Entries`);
}

/* Playback */
async function startPlay() {
  if (running || !trained) return;
  running=true; stopFlag=false;
  document.getElementById('playBtn').disabled  = true;
  document.getElementById('trainBtn').disabled = true;
  clearLog();
  const envMode = modules.randomenv ? 'New Random Map' : 'Static Map';
  log(`<span class="rl"># Trained Policy — ${envMode}</span>`);

  const result = runEpisode(0, true);

  for (let i=0; i<result.frames.length && !stopFlag; i++) {
    const f = result.frames[i];
    drawState(f.rl, f.goap, f.path, f.pounceTrail, f.resting, f.exhausted, f.gridSnapshot, f.foodSnapshot);
    if (modules.rest)   updateStaminaBar(f.sprintCount, f.restCount);
    if (modules.energy) updateEnergyBar(f.energy);
    if (f.caught) { log(`  <span class="err">Caught at Move ${f.moves}</span>`); break; }
    await delay(90);
  }
  if (result.survived) log(`<span class="ok">  Survived all ${MAX_MOVES} Moves!</span>`);
  running=false;
  log(`  Moves: ${result.moves}  Survived: ${result.survived}`);
  document.getElementById('playBtn').disabled  = false;
  document.getElementById('trainBtn').disabled = false;
}

/* Reset */
function hardReset() {
  stopFlag=true; running=false; trained=false;
  qtable={}; rewardHistory=[];
  makeGrid();
  drawState(RL_START, GOAP_START, [], [], false, false, staticGrid, staticFood);
  updateEnergyBar(MAX_ENERGY);
  clearLog();
  document.getElementById('prog').style.width     = '0%';
  document.getElementById('prog-label').textContent = '';
  cctx.clearRect(0, 0, chartCanvas.width, chartCanvas.height);
  updateStaminaBar(0, 0);
  document.getElementById('rest-label').textContent = '';
  document.getElementById('trainBtn').disabled = false;
  document.getElementById('playBtn').disabled  = true;
  document.getElementById('trainBtn').textContent = `Train RL (${getTrainEpisodes()} Runs)`;
  const d     = document.getElementById('density-slider').value;
  const wsize = windowHalf*2+1;
  log(`<span class="kw"># Reset — Walls ${d}%  Window ${wsize}×${wsize}  Food (${staticFood.x},${staticFood.y})  Env: ${modules.randomenv?'Randomized':'Fixed'}</span>`);
}

/* Boot */
makeGrid();
drawState(RL_START, GOAP_START, [], [], false, false, staticGrid, staticFood);
updateEnergyBar(MAX_ENERGY);
updateStaminaBar(0, 0);
document.getElementById('trainBtn').textContent = `Train RL (${getTrainEpisodes()} Runs)`;
log(`<span class="kw"># GOAP vs RL </span>`);
log(`  Configure Modules Above, press <span class="kw">Reset</span> to Load Module Changes, Press <span class="kw">Train RL</span> to Train`);
