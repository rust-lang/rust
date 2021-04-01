// compile-flags: --no-source

#![crate_name = "foo"]

// @!has src/foo/html_no_source_attr.rs.html
// @has foo/index.html
// @!has - '//*[@class="srclink"]' '[src]'
