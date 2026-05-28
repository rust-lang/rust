//@ aux-build:submodule-inner.rs
//@ build-aux-docs
#![deny(rustdoc::broken_intra_doc_links)]

extern crate a;

//@ has 'submodule_inner/struct.Foo.html' '//a[@href="../a/bar/struct.Bar.html"]' 'Bar'
pub use a::foo::Foo;
