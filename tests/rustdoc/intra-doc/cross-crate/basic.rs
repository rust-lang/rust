// aux-build:intra-doc-basic.rs
// build-aux-docs
#![deny(rustdoc::broken_intra_doc_links)]

// from https://github.com/rust-lang/rust/issues/65983
extern crate a;

// @has 'basic/struct.Bar.html' '//a[@href="../a/struct.Foo.html"]' 'Foo'
pub use a::Bar;
