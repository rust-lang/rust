// Issue #876

use std::vec::Vec;

fn last<T>(v: Vec<&T> ) -> std::option::Option<T> {
    ::std::panic!();
}

fn main() {
    let y;
    let x : char = last(y);
    //~^ ERROR mismatched types
    //~| NOTE_NONVIRAL expected type `char`
    //~| NOTE_NONVIRAL found enum `Option<_>`
    //~| NOTE_NONVIRAL expected `char`, found `Option<_>`
}
