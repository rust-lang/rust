// Regression test for the ICE described in #82328. The pretty-printer for
// `-Zunpretty=hir,typed` would previously retrieve type-checking results
// when entering a body, which means that type information was not available
// for expressions occurring in function signatures, as in the `foo` example
// below, leading to an ICE.

//@ check-pass
//@ compile-flags: -Zunpretty=hir,typed
//@ edition: 2015
#![allow(dead_code)]

fn main() {}

fn foo(-128..=127: i8) {}
