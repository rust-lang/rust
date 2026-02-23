//@ run-pass
//@ check-run-results
//@ aux-build: other_crate_privacy1.rs
//@ ignore-backends: gcc
// FIXME: linking on windows (specifically mingw) not yet supported, see tracking issue #125418
//@ ignore-windows
// Tests whether re-exports work.
#![feature(extern_item_impls)]

extern crate other_crate_privacy1 as codegen;

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
