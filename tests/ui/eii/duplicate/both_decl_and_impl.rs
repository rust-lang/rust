//@ ignore-backends: gcc
// FIXME(#125418): linking on Windows GNU targets is not yet supported.
//@ ignore-windows-gnu
// Tests that one item can't both define and impl an EII at the same time
#![feature(extern_item_impls)]

#[eii]
fn a(x: u64);

#[a]
#[eii]
//~^ ERROR a single item cannot both declare and implement EIIs
fn b(x: u64) {}

#[eii]
fn c(x: u64);
//~^ ERROR `#[c]` function required, but not found

#[eii]
#[c]
fn d(x: u64) {}
//~^ ERROR only a small subset of attributes are supported on externally implementable items

fn main() {
    a(42);
    b(42);
}
