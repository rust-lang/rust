// Test that we use fully-qualified type names in error messages.

import core::task::Task;

fn bar(x: uint) -> Task {
    return x;
    //~^ ERROR mismatched types: expected `core::task::Task`
}

fn main() {
}
