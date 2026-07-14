//@ run-pass
//@ check-run-results
//@ aux-build: cross_crate_decl.rs
//@ compile-flags: -O
//@ ignore-backends: gcc
// FIXME(#125418): linking on Windows GNU targets is not yet supported.
//@ ignore-windows-gnu
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
