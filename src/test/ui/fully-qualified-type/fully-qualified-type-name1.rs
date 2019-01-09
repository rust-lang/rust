// Test that we use fully-qualified type names in error messages.

fn main() {
    let x: Option<usize>;
    x = 5;
    //~^ ERROR mismatched types
    //~| expected type `std::option::Option<usize>`
    //~| found type `{integer}`
    //~| expected enum `std::option::Option`, found integer
}
