// aux-build:external-macro-src.rs
// ignore-tidy-linelength

#![crate_name = "foo"]

#[macro_use]
extern crate external_macro_src;

// @has foo/index.html '//a[@href="../src/foo/external-macro-src.rs.html#4-15"]' '[src]'

// @has foo/struct.Foo.html
// @has - '//a[@href="https://example.com/src/external_macro_src/external-macro-src.rs.html#8"]' '[src]'
// @has - '//a[@href="https://example.com/src/external_macro_src/external-macro-src.rs.html#9-13"]' '[src]'
// @has - '//a[@href="https://example.com/src/external_macro_src/external-macro-src.rs.html#10-12"]' '[src]'
make_foo!();
