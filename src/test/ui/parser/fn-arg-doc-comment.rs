pub fn f(
    /// Comment
    //~^ ERROR documentation comments cannot be applied to function parameters
    //~| NOTE doc comments are not allowed here
    id: u8,
    /// Other
    //~^ ERROR documentation comments cannot be applied to function parameters
    //~| NOTE doc comments are not allowed here
    a: u8,
) {}

fn bar(id: #[allow(dead_code)] i32) {}
//~^ ERROR attributes cannot be applied to a function parameter's type
//~| NOTE attributes are not allowed here

fn main() {
    // verify that the parser recovered and properly typechecked the args
    f("", "");
    //~^ ERROR mismatched types
    //~| NOTE expected `u8`, found `&str`
    //~| ERROR mismatched types
    //~| NOTE expected `u8`, found `&str`
    bar("");
    //~^ ERROR mismatched types
    //~| NOTE expected `i32`, found `&str`
}
