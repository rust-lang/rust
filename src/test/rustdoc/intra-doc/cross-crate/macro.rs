// aux-build:macro_inner.rs
// aux-build:proc_macro.rs
// build-aux-docs
#![deny(rustdoc::broken_intra_doc_links)]
extern crate macro_inner;
extern crate proc_macro_inner;

// @has 'macro/macro.my_macro.html' '//a[@href="../macro_inner/struct.Foo.html"]' 'Foo'
pub use macro_inner::my_macro;
// @has 'macro/derive.DeriveA.html' '//a[@href="../proc_macro_inner/derive.OtherDerive.html"]' 'OtherDerive'
pub use proc_macro_inner::DeriveA;
