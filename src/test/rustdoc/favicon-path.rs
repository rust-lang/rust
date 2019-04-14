// compile-flags:-Z unstable-options --favicon-path ./src/librustdoc/html/static/favicon.ico

#![crate_name = "foo"]

// @has foo/fn.foo.html '//link/@href' '../favicon-foo.ico'
pub fn foo() {}
