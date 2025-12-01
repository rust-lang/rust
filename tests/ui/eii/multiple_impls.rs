//@ run-pass
//@ check-run-results
//@ ignore-backends: gcc
// Tests whether one function could implement two EIIs.
#![feature(extern_item_impls)]

#[eii]
fn a(x: u64);

#[eii]
fn b(x: u64);

#[a]
#[b]
fn implementation(x: u64) {
    println!("{x:?}")
}

// what you would write:
fn main() {
    a(42);
    b(42);
}
