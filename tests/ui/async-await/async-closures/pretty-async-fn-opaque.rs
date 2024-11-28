//@ edition: 2021

#![feature(async_closure)]

use std::ops::AsyncFnMut;

fn produce() -> impl AsyncFnMut() -> &'static str {
    async || ""
}

fn main() {
    let x: i32 = produce();
    //~^ ERROR mismatched types
}
