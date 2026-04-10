//@ run-pass
//@ ignore-backends: gcc
//@ ignore-windows

#![feature(extern_item_impls)]
#![feature(contracts)]
#![allow(incomplete_features)]

#[eii(hello)]
fn hello(x: u64);

#[hello]
#[core::contracts::requires(x > 0)]
fn hello_impl(x: u64) {
    println!("{x:?}")
}

fn main() {
    hello(42);
}
