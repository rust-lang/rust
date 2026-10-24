//! Regression test for <https://github.com/rust-lang/rust/issues/98002>.
//!
//! Keywords should not be generated in rustdoc JSON output and this test
//! ensures it.

#![feature(rustdoc_internals)]
#![no_std]

//@ !has "$.index[?(@.name=='match')]"
#[doc(keyword = "match")]
/// this is a test!
const _: () = ();

//@ !has "$.index[?(@.name=='break')]"
#[doc(keyword = "break")]
/// hello
const _: () = ();
