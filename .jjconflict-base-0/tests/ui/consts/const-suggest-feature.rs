//@ edition: 2018
use std::cell::Cell;

const WRITE: () = unsafe {
    let x = async { 13 };
    //~^ ERROR `async` blocks
    //~| HELP add `#![feature(const_async_blocks)]` to the crate attributes to enable
};

fn main() {}
