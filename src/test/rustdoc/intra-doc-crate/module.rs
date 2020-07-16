// outer.rs
// aux-build: module.rs
// build-aux-docs
#![deny(intra_doc_link_resolution_failure)]
extern crate module_inner;
// @has 'module/bar/index.html' '//a[@href="../../module_inner/trait.SomeTrait.html"]' 'SomeTrait'
// @has 'module/bar/index.html' '//a[@href="../../module_inner/struct.SomeType.html"]' 'SomeType'
pub use module_inner::bar;
