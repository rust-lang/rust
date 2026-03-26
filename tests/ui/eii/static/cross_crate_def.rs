//@ run-pass
//@ check-run-results
//@ aux-build: cross_crate_def.rs
//@ compile-flags: -O
//@ ignore-backends: gcc
// FIXME: linking on windows (speciifcally mingw) not yet supported, see tracking issue #125418
//@ ignore-windows
// Tests whether calling EIIs works with the declaration and definition in another crate.

extern crate cross_crate_def as codegen;

// what you would write:
fn main() {
    // directly
    println!("{}", codegen::EII1_IMPL);
    // through the alias
    println!("{}", codegen::DECL1);
}
