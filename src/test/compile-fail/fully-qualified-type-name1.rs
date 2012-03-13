// Test that we use fully-qualified type names in error messages.

fn main() {
    let x: option<uint>;
    x = 5;
    //!^ ERROR mismatched types: expected `core::option::option<uint>`
}
