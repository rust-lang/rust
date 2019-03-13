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
    //~| found type `std::option::Option<_>`
    //~| expected char, found enum `std::option::Option`
}
