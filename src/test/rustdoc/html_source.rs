#![crate_name = "foo"]

// @has src/foo/html_source.rs.html
// @has foo/index.html
// @has - '//*[@class="srclink"]' '[src]'
