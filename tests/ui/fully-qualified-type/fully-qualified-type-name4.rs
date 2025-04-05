// Test that we use fully-qualified type names in error messages.

use std::option::Option;

fn bar(x: usize) -> Option<usize> {
    return x;
    //~^ ERROR mismatched types
    //~| NOTE_NONVIRAL expected enum `Option<usize>`
    //~| NOTE_NONVIRAL found type `usize`
    //~| NOTE_NONVIRAL expected `Option<usize>`, found `usize`
}

fn main() {
}
