// outer.rs
//@ aux-build: module.rs
//@ build-aux-docs
#![deny(rustdoc::broken_intra_doc_links)]
extern crate module_inner;
//@ has 'module/bar/index.html' '//a[@href="../../module_inner/trait.SomeTrait.html"]' 'SomeTrait'
//@ has 'module/bar/index.html' '//a[@href="../../module_inner/struct.SomeType.html"]' 'SomeType'
pub use module_inner::bar;
