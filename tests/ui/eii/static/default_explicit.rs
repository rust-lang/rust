//@ run-pass
//@ check-run-results
//@ ignore-backends: gcc
// FIXME(#125418): linking on Windows GNU targets is not yet supported.
//@ ignore-windows-gnu
// FIXME(#157649): static EII defaults currently fail to link on Apple targets.
//@ ignore-apple
// Tests that an explicit static EII implementation overrides a local default.

#![feature(extern_item_impls)]
#![allow(dead_code)]

#[eii(eii1)]
pub static DECL1: u64 = 5;

#[eii1]
pub static EII1_IMPL: u64 = 10;

fn main() {
    println!("{DECL1}");
}
