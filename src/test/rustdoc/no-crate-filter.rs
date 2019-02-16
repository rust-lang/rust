#![crate_name = "foo"]

// compile-flags: -Z unstable-options --disable-per-crate-search

// @!has 'foo/struct.Foo.html' '//*[id="crate-search"]'
pub struct Foo;
