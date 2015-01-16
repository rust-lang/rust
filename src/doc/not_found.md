% Not Found

<!-- Completely hide the TOC and the section numbers -->
<style type="text/css">
#TOC { display: none; }
.header-section-number { display: none; }
li {list-style-type: none; }
</style>

Looks like you've taken a wrong turn.

Some things that might be helpful to you though:

## Search

* <form action="https://duckduckgo.com/">
    <input type="text" id="site-search" name="q" size="80"></input>
    <input type="submit" value="Search DuckDuckGo">
</form>
* Rust doc search: <span id="core-search"></span>

## Reference

* [The Rust official site](http://rust-lang.org)
* [The Rust reference](http://doc.rust-lang.org/reference.html)

## Docs

* [The standard library](http://doc.rust-lang.org/std/)

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
    var lt = op.pop();

    // #18540, use a single token

    var search = document.getElementById('core-search');
    search.innerHTML = "<a href=\"http://doc.rust-lang.org/core/?search=" + lt + "\">" + lt + "</a>";
}
populate_site_search();
populate_rust_search();
</script>

