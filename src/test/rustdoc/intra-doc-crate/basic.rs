// aux-build:intra-doc-basic.rs
// build-aux-docs
#![deny(intra_doc_resolution_failures)]

// from https://github.com/rust-lang/rust/issues/65983
extern crate a;

// @has 'basic/struct.Bar.html' '//a[@href="../a/struct.Foo.html"]' 'Foo'
pub use a::Bar;
