var shardCache = {};
var shardPromises = {};

function loadSource(details) {
  if (!details.open) return;
  if (details.getAttribute('data-loaded') === '1') return;
  var fnId = details.getAttribute('data-fn-id');
  var body = details.querySelector('.src-body');
  var shardKey = Number(fnId) % SHARD_COUNT;

  var fetchPromise = shardPromises[shardKey];
  if (!fetchPromise) {
    fetchPromise = fetch(SHARD_DIR + '/shard-' + shardKey + '.json')
      .then(r => r.json())
      .then(data => { shardCache[shardKey] = data; return data; });
    shardPromises[shardKey] = fetchPromise;
  }

  fetchPromise.then(data => {
    var lines = data[fnId];
    if (!lines) {
      body.innerHTML = '<tr><td class="code">(source unavailable)</td></tr>';
      return;
    }
    var classMap = { c: 'line-covered', u: 'line-uncovered', i: 'line-ignored' };
    var html = '';
    for (var i = 0; i < lines.length; i++) {
      var ln = lines[i];
      var cls = classMap[ln.c] || 'line-ignored';
      html += '<tr class="' + cls + '">'
        + '<td class="lineno">' + ln.n + '</td>'
        + '<td class="code">' + escapeHtml(ln.t) + '</td></tr>';
    }
    body.innerHTML = html;
    details.setAttribute('data-loaded', '1');
  }).catch(err => {
    body.innerHTML = '<tr><td class="code">(failed to load source: '
      + escapeHtml(String(err)) + ')</td></tr>';
  });
}

function escapeHtml(s) {
  var div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}
