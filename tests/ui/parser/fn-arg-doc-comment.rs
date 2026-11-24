pub fn f( //~ NOTE function defined here
    /// Comment
    //~^ ERROR unused doc comment
    //~| NOTE rustdoc does not generate documentation for function params
    id: u8,
    /// Other
    //~^ ERROR unused doc comment
    //~| NOTE rustdoc does not generate documentation for function params
    a: u8,
) {}

fn bar(id: #[allow(dead_code)] i32) {}
//~^ ERROR attributes cannot be applied to types
//~| NOTE attributes are not allowed here
//~| NOTE function defined here

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
