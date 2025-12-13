//@ run-pass
//@ check-run-results
//@ ignore-backends: gcc
// FIXME: linking on windows (speciifcally mingw) not yet supported, see tracking issue #125418
//@ ignore-windows
// Tests EIIs with default implementations.
// In the same crate, the explicit implementation should get priority.
#![feature(extern_item_impls)]

#[eii(eii1)]
pub fn decl1(x: u64) {
    //~^ WARN function `decl1` is never used
    println!("default {x}");
}

#[eii1]
pub fn decl2(x: u64) {
    println!("explicit {x}");
}

fn main() {
    decl1(4);
}
