% Not Found

<!-- Completely hide the TOC and the section numbers -->
<style type="text/css">
#TOC { display: none; }
.header-section-number { display: none; }
li {list-style-type: none; }
.search-input {
    width: calc(100% - 200px);
}
.search-but {
    cursor: pointer;
}
.search-but, .search-input {
    padding: 4px;
    border: 1px solid #ccc;
    border-radius: 3px;
    outline: none;
    font-size: 0.7em;
    background-color: #fff;
}
.search-but:hover, .search-input:focus {
    border-color: #55a9ff;
}
</style>

Looks like you've taken a wrong turn.

Some things that might be helpful to you though:

# Search

<div>
  <form action="std/index.html" method="get">
    <input id="std-search" class="search-input" type="search" name="search"
           placeholder="Search through the standard library"/>
    <button class="search-but">Search Standard Library</button>
  </form>
</div>

<div>
  <form action="https://duckduckgo.com/">
    <input id="site-search" class="search-input" type="search" name="q"></input>
    <input type="submit" value="Search DuckDuckGo" class="search-but">
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

function populate_site_search() {
    var op = get_url_fragments();

    var search = document.getElementById('site-search');
    search.value = op.join(' ') + " site:doc.rust-lang.org";
}

function populate_rust_search() {
    var op = get_url_fragments();
    document.getElementById('std-search').value = op.join(' ');
}
populate_site_search();
populate_rust_search();
</script>
