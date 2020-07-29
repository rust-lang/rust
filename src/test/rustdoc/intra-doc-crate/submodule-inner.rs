// aux-build:submodule-inner.rs
// build-aux-docs
#![deny(intra_doc_resolution_failures)]

extern crate a;

// @has 'submodule_inner/struct.Foo.html' '//a[@href="../a/bar/struct.Bar.html"]' 'Bar'
pub use a::foo::Foo;
