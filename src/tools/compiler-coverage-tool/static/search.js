var currentSearch = '';

function applyFilters() {
  var query = currentSearch.toLowerCase();
  var visible = 0;
  document.querySelectorAll('.fn-block').forEach(el => {
    var nameEl = el.querySelector('.fn-name');
    var fileEl = el.querySelector('.fn-file');
    var name = nameEl ? nameEl.textContent.toLowerCase() : '';
    var file = fileEl ? fileEl.textContent.toLowerCase() : '';
    var hide = query !== '' && !name.includes(query) && !file.includes(query);
    el.classList.toggle('hidden', hide);
    if (!hide) visible++;
  });
  document.querySelectorAll('.file-group').forEach(el => {
    el.style.display = el.querySelector('.fn-block:not(.hidden)') ? '' : 'none';
  });
  document.querySelectorAll('.crate-group').forEach(el => {
    el.style.display = el.querySelector('.fn-block:not(.hidden)') ? '' : 'none';
  });
  var countEl = document.getElementById('search-count');
  if (countEl) countEl.textContent = query ? visible + ' result' + (visible === 1 ? '' : 's') : '';
}

function onSearch(val) {
  currentSearch = val;
  applyFilters();
}
