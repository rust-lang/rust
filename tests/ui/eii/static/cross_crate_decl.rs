//@ run-pass
//@ check-run-results
//@ aux-build: cross_crate_decl.rs
//@ compile-flags: -O
//@ ignore-backends: gcc
// FIXME: linking on windows (speciifcally mingw) not yet supported, see tracking issue #125418
//@ ignore-windows
// Tests whether calling EIIs works with the declaration in another crate.

extern crate cross_crate_decl as codegen;

#[codegen::eii1]
static EII1_IMPL: u64 = 5;

// what you would write:
fn main() {
    // directly
    println!("{}", EII1_IMPL);
    // through the alias
    println!("{}", codegen::DECL1);
}
