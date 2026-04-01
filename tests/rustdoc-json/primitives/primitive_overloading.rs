//@ compile-flags: --document-private-items

// Regression test for <https://github.com/rust-lang/rust/issues/98006>.

#![feature(rustc_attrs)]

//@ has "$.index[?(@.name=='usize')]"
//@ has "$.index[?(@.name=='prim')]"

#[rustc_doc_primitive = "usize"]
/// This is the built-in type `usize`.
mod prim {}
