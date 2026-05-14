//@ run-pass
//@ check-run-results
//@ ignore-backends: gcc
// FIXME: linking on windows (specifically mingw) not yet supported, see tracking issue #125418
//@ ignore-windows
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
