//@ run-pass
//@ check-run-results
//@ ignore-backends: gcc
// Tests whether calling EIIs works with the declaration in the same crate.
#![feature(extern_item_impls)]

#[eii]
fn hello(x: u64);

#[hello]
fn hello_impl(x: u64) {
    println!("{x:?}")
}

// what you would write:
fn main() {
    // directly
    hello_impl(21);
    // through the alias
    hello(42);
}
