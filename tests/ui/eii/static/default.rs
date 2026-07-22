//@ run-pass
//@ check-run-results
//@ ignore-backends: gcc
// FIXME(#125418): linking on Windows GNU targets is not yet supported.
//@ ignore-windows-gnu
// FIXME(#157649): static EII defaults currently fail to link on Apple targets.
//@ ignore-apple
// Tests static EIIs with default implementations.

#![feature(extern_item_impls)]

#[eii(eii1)]
pub static DECL1: u64 = 5;

fn main() {
    println!("{DECL1}");
}
