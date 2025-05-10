//@ run-pass
//@ check-run-results
//@ aux-build: codegen1.rs
#![feature(eii)]

extern crate codegen1 as codegen;

#[codegen::eii1]
fn eii1_impl(x: u64) {
    println!("{x:?}")
}


#[codegen::eii3]
fn eii3_impl(x: u64) {
    println!("{x:?}")
}

// what you would write:
fn main() {
    // directly
    eii1_impl(21);
    // through the alias
    codegen::local_call_decl1(42);

    // directly
    eii3_impl(12);
    // through the alias
    codegen::decl3(24);
}
