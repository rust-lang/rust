// Test that we use fully-qualified type names in error messages.

use std::option::Option;

fn bar(x: usize) -> Option<usize> {
    return x;
    //~^ ERROR mismatched types
    //~| expected type `std::option::Option<usize>`
    //~| found type `usize`
    //~| expected enum `std::option::Option`, found usize
}

fn main() {
}
