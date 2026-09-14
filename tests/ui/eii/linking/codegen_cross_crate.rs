//@ run-pass
//@ check-run-results
//@ aux-build: codegen_cross_crate_other_crate.rs
//@ compile-flags: -O
//@ ignore-backends: gcc
// FIXME(#125418): linking on Windows GNU targets is not yet supported.
//@ ignore-windows-gnu
// Tests whether calling EIIs works with the declaration in another crate.

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
