// aux-build:traits.rs
// build-aux-docs
// ignore-tidy-line-length
#![deny(intra_doc_link_resolution_failure)]

extern crate inner;
use inner::SomeTrait;

pub struct SomeStruct;

 // @has 'traits/struct.SomeStruct.html' '//a[@href="../inner/trait.SomeTrait.html"]' 'SomeTrait'
impl SomeTrait for SomeStruct {
    // @has 'traits/struct.SomeStruct.html' '//a[@href="../inner/trait.SomeTrait.html"]' 'SomeTrait'
    fn foo() {}
}
