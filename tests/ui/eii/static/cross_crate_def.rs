//@ run-pass
//@ check-run-results
//@ aux-build: cross_crate_def.rs
//@ compile-flags: -O
//@ ignore-backends: gcc
// FIXME(#125418): linking on Windows GNU targets is not yet supported.
//@ ignore-windows-gnu
// Tests whether calling EIIs works with the declaration and definition in another crate.

extern crate cross_crate_def as codegen;

// what you would write:
fn main() {
    // directly
    println!("{}", codegen::EII1_IMPL);
    // through the alias
    println!("{}", codegen::DECL1);
}
