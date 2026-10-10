// Doc attributes (`#[doc(attribute = "...")]` should not be generated in rustdoc JSON output
// and this test ensures it.

#![feature(rustdoc_internals)]
#![no_std]

//@ !has "$.index[?(@.name=='repr')]"
#[doc(attribute = "repr")]
/// this is a test!
const _: () = ();

//@ !has "$.index[?(@.name=='forbid')]"
#[doc(attribute = "forbid")]
/// hello
const _: () = ();
