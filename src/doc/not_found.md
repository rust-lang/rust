% Not Found

<!-- Completely hide the TOC and the section numbers -->
<style type="text/css">
#rustdoc-toc { display: none; }
.header-section-number { display: none; }
li {list-style-type: none; }
#search-input {
    width: calc(100% - 100px);
}
#search-but {
    cursor: pointer;
}
#search-but, #search-input {
    padding: 4px;
    border: 1px solid #ccc;
    border-radius: 3px;
    outline: none;
    font-size: 0.7em;
    background-color: #fff;
}
#search-but:hover, #search-input:focus {
    border-color: #55a9ff;
}
#search-from {
    border: none;
    padding: 0;
    font-size: 0.7em;
}
</style>

Looks like you've taken a wrong turn.

Some things that might be helpful to you though:

# Search

<div>
  <form id="search-form" action="https://duckduckgo.com/">
    <input id="search-input" type="search" name="q"></input>
    <input type="submit" value="Search" id="search-but">
    <!--
      Don't show the options by default,
      since "From the Standary Library" doesn't work without JavaScript
    -->
    <fieldset id="search-from" style="display:none">
      <label><input name="from" value="library" type="radio"> From the Standard Library</label>
      <label><input name="from" value="duckduckgo" type="radio" checked> From DuckDuckGo</label>
    </fieldset>
  </form>
</div>

# Reference

 * [The Rust official site](https://www.rust-lang.org)
 * [The Rust reference](https://doc.rust-lang.org/reference/index.html)

# Docs

[The standard library](https://doc.rust-lang.org/std/)

<script>
function get_url_fragments() {
    var last = document.URL.split("/").pop();
    var tokens = last.split(".");
    var op = [];
    for (var i=0; i < tokens.length; i++) {
        var t = tokens[i];
        if (t == 'html' || t.indexOf("#") != -1) {
            // no html or anchors
        } else {
            op.push(t);
        }
    }
    return op;
}

function on_submit(event) {
    var form = event.target;
    var q = form['q'].value;

    event.preventDefault();

    if (form['from'].value === 'duckduckgo') {
        document.location.href = form.action + '?q=' + encodeURIComponent(q + ' site:doc.rust-lang.org');
    } else if (form['from'].value === 'library') {
        document.location.href = '/std/index.html?search=' + encodeURIComponent(q);
    }
}

function populate_search() {
    var form = document.getElementById('search-form');
    form.addEventListener('submit', on_submit);
    document.getElementById('search-from').style.display = '';

    form['from'].value = 'library';

    var op = get_url_fragments();
    document.getElementById('search-input').value = op.join(' ');
}
populate_search();
</script>
