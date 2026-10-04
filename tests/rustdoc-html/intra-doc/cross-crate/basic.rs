//@ aux-build:intra-doc-basic.rs
//@ build-aux-docs
#![deny(rustdoc::broken_intra_doc_links)]

// from https://github.com/rust-lang/rust/issues/65983
extern crate a;

//@ has 'basic/type.Bar.html' '//a[@href="../a/type.Foo.html"]' 'Foo'
pub use a::Bar;
