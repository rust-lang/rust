// Regression test for <https://github.com/rust-lang/rust/issues/98002>.

// Keywords should not be generated in rustdoc JSON output and this test
// ensures it.

#![feature(rustdoc_internals)]
#![no_std]

//@ !has "$.index[?(@.name=='match')]"
//@ has "$.index[?(@.name=='foo')]"

#[doc(keyword = "match")]
/// this is a test!
pub mod foo {}

//@ !has "$.index[?(@.name=='break')]"
//@ !has "$.index[?(@.name=='bar')]"
#[doc(keyword = "break")]
/// hello
mod bar {}
