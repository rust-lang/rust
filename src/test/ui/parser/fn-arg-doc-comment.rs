pub fn f( //~ NOTE function defined here
    /// Comment
    //~^ ERROR documentation comments cannot be applied to function parameters
    //~| NOTE doc comments are not allowed here
    //~| NOTE
    id: u8,
    /// Other
    //~^ ERROR documentation comments cannot be applied to function parameters
    //~| NOTE doc comments are not allowed here
    //~| NOTE
    a: u8,
) {}

fn bar(id: #[allow(dead_code)] i32) {}
//~^ ERROR attributes cannot be applied to a function parameter's type
//~| NOTE attributes are not allowed here
//~| NOTE function defined here
//~| NOTE

fn main() {
    // verify that the parser recovered and properly typechecked the args
    f("", "");
    //~^ ERROR arguments to this function are incorrect
    //~| NOTE expected `u8`, found `&str`
    //~| NOTE expected `u8`, found `&str`
    bar("");
    //~^ ERROR mismatched types
    //~| NOTE arguments to this function are incorrect
    //~| NOTE expected `i32`, found `&str`
}
