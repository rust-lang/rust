//@ run-pass
//@ check-run-results
//@ ignore-backends: gcc
// FIXME: linking on windows (specifically mingw) not yet supported, see tracking issue #125418
//@ ignore-windows
// FIXME(#157649): static EII defaults currently fail to link on Apple targets.
//@ ignore-apple
// Tests static EIIs with default implementations.

#![feature(extern_item_impls)]

#[eii(eii1)]
pub static DECL1: u64 = 5;

fn main() {
    println!("{DECL1}");
}
