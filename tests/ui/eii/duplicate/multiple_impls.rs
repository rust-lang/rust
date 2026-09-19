//@ ignore-backends: gcc
// FIXME(#125418): linking on Windows GNU targets is not yet supported.
//@ ignore-windows-gnu
// Tests that one item can't implement two EIIs
#![feature(extern_item_impls)]

#[eii]
fn a(x: u64);
//~^ ERROR `#[a]` function required, but not found

#[eii]
fn b(x: u64);

#[a]
#[b]
//~^ ERROR a single item cannot implement multiple EIIs
fn implementation(x: u64) {
    println!("{x:?}")
}

#[eii(c)]
//~^ ERROR `#[c]` static required, but not found
static C: u64;

#[eii(d)]
static D: u64;

#[c]
#[d]
//~^ ERROR a single item cannot implement multiple EIIs
static IMPL: u64 = 5;

fn main() {
    a(42);
    b(42);
    println!("{C} {D} {IMPL}")
}
