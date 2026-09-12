//@ run-pass
//@ check-run-results
//@ aux-build: other_crate_privacy1.rs
//@ ignore-backends: gcc
// FIXME(#125418): linking on Windows GNU targets is not yet supported.
//@ ignore-windows-gnu
// Tests whether re-exports work.

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
