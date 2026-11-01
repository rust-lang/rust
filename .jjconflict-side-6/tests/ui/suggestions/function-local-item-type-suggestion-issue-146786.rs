//@ run-rustfix
#![allow(dead_code)]

fn main() {
    struct Error;

    const ERROR = Error;
    //~^ ERROR missing type for `const` item
    //~| HELP provide a type for the constant
    //~| SUGGESTION : Error
}
