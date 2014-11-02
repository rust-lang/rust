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
    <input type="text" id="code.search" name="q" size="80"></input>
    <input type="submit" value="Search DuckDuckGo">
</form>

## Reference
* [The Rust official site](http://rust-lang.org)
* [The Rust reference](http://doc.rust-lang.org/reference.html) (* [PDF](http://doc.rust-lang.org/reference.pdf))

## Docs
* [The standard library](http://doc.rust-lang.org/std/)

<script>
function populate_search_box() {

    var last = document.URL.split("/").pop();
    var tokens = last.split(".");
    var op = [];
    for (var i=0; i < tokens.length; i++) {
        if (tokens[i].indexOf("#") == -1) 
            op.push(tokens[i]);
    }

    var search = document.getElementById('code.search');
    search.value = op.join(' ') + " site:doc.rust-lang.org";
}
populate_search_box();
</script>

