// Issue #876

use std::vec::Vec;

fn last<T>(v: Vec<&T> ) -> std::option::Option<T> {
    ::std::panic!();
}

fn main() {
    let y;
    let x : char = last(y);
    //~^ ERROR mismatched types
    //~| expected type `char`
    //~| found enum `Option<_>`
    //~| expected `char`, found `Option<_>`
}
