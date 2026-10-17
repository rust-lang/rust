// Issue #876

use std::vec::Vec;

fn last<T>(v: Vec<&T> ) -> std::option::Option<T> {
    ::std::panic!();
}

fn main() {
    let y;
    let x : char = last(y);
    //~^ ERROR mismatched types
    //~| NOTE expected type `char`
    //~| NOTE found enum `Option<_>`
    //~| NOTE expected `char`, found `Option<_>`
    //~| NOTE expected due to this
}
