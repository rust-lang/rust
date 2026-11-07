#![allow(non_camel_case_types)]

struct bool; //~ NOTE the other `bool` is defined in the current crate
struct str; //~ NOTE the other `str` is defined in the current crate

fn foo(_: bool) {} //~ NOTE function defined here
fn bar(_: &str) {} //~ NOTE function defined here

fn main() {
    foo(true);
    //~^ ERROR mismatched types [E0308]
    //~| NOTE expected `bool`, found a different `bool`
    //~| NOTE arguments to this function are incorrect
    //~| NOTE `bool` and `bool` have similar names, but are actually distinct types
    //~| NOTE one `bool` is a primitive defined by the language
    bar("hello");
    //~^ ERROR mismatched types [E0308]
    //~| NOTE expected `str`, found a different `str`
    //~| NOTE arguments to this function are incorrect
    //~| NOTE `str` and `str` have similar names, but are actually distinct types
    //~| NOTE one `str` is a primitive defined by the language
}
