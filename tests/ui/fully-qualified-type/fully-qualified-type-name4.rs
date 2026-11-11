// Test that we use fully-qualified type names in error messages.

use std::option::Option;

fn bar(x: usize) -> Option<usize> { //~ NOTE expected `Option<usize>` because of return type
    return x;
    //~^ ERROR mismatched types
    //~| NOTE expected enum `Option<usize>`
    //~| NOTE found type `usize`
    //~| NOTE expected `Option<usize>`, found `usize`
}

fn main() {
}
