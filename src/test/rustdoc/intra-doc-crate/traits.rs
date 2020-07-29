// ignore-test
// ^ this is https://github.com/rust-lang/rust/issues/73829
// aux-build:traits.rs
// build-aux-docs
// ignore-tidy-line-length
#![deny(intra_doc_resolution_failures)]

extern crate inner;
use inner::SomeTrait;

pub struct SomeStruct;

 // @has 'traits/struct.SomeStruct.html' '//a[@href="../inner/trait.SomeTrait.html"]' 'SomeTrait'
impl SomeTrait for SomeStruct {
    // @has 'traits/struct.SomeStruct.html' '//a[@href="../inner/trait.SomeTrait.html"]' 'a trait'
    fn foo() {}
}
