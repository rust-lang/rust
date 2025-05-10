//@ run-pass
//@ check-run-results
//@ aux-build: codegen2.rs
//@ compile-flags: -O
#![feature(eii)]

extern crate codegen2 as codegen;

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
