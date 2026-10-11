//@ aux-build:submodule-inner.rs
//@ build-aux-docs
#![deny(rustdoc::broken_intra_doc_links)]

extern crate a;

//@ has 'submodule_inner/type.Foo.html' '//a[@href="../a/bar/type.Bar.html"]' 'Bar'
pub use a::foo::Foo;
