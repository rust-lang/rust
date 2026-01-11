//@ run-pass
//@ check-run-results
//@ aux-build: codegen_cross_crate_other_crate.rs
//@ compile-flags: -O
//@ ignore-backends: gcc
// FIXME: linking on windows (speciifcally mingw) not yet supported, see tracking issue #125418
//@ ignore-windows
// Tests whether calling EIIs works with the declaration in another crate.
#![feature(extern_item_impls)]

extern crate codegen_cross_crate_other_crate as codegen;

#[codegen::eii1]
fn eii1_impl(x: u64) {
    println!("{x:?}")
}

// what you would write:
fn main() {
    // directly
    eii1_impl(21);
    // through the alias
    codegen::decl1(42);
}
