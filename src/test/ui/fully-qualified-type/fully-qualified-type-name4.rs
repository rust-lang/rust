// Test that we use fully-qualified type names in error messages.

use std::option::Option;

fn bar(x: usize) -> Option<usize> {
    return x;
    //~^ ERROR mismatched types
    //~| expected enum `Option<usize>`
    //~| found type `usize`
    //~| expected enum `Option`, found `usize`
}

fn main() {
}
