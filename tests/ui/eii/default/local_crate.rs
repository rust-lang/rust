//@ run-pass
//@ check-run-results
//@ ignore-backends: gcc
// FIXME: linking on windows (speciifcally mingw) not yet supported, see tracking issue #125418
//@ ignore-windows
// Tests EIIs with default implementations.
// In the same crate, when there's no explicit declaration, the default should be called.
#![feature(extern_item_impls)]

#[eii(eii1)]
pub fn decl1(x: u64) {
    println!("default {x}");
}

fn main() {
    decl1(4);
}
