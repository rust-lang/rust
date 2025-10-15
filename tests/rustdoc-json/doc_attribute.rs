// Doc attributes (`#[doc(attribute = "...")]` should not be generated in rustdoc JSON output
// and this test ensures it.

#![feature(rustdoc_internals)]
#![no_std]

//@ !has "$.index[?(@.name=='repr')]"
//@ has "$.index[?(@.name=='foo')]"

#[doc(attribute = "repr")]
/// this is a test!
pub mod foo {}

//@ !has "$.index[?(@.name=='forbid')]"
//@ !has "$.index[?(@.name=='bar')]"
#[doc(attribute = "forbid")]
/// hello
mod bar {}
