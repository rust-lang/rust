//@ run-pass
//@ check-run-results
//@ ignore-backends: gcc
// FIXME: linking on windows (specifically mingw) not yet supported, see tracking issue #125418
//@ ignore-windows
// Tests static EIIs with default implementations.

#![feature(extern_item_impls)]

#[eii(eii1)]
pub static DECL1: u64 = 5;

fn main() {
    println!("{DECL1}");
}
