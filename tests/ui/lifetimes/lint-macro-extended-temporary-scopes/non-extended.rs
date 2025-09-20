//! Test that `macro_extended_temporary_scopes` doesn't warn on non-extended temporaries.
//@ edition: 2024
#![deny(macro_extended_temporary_scopes)] //~ WARN unknown lint

fn temp() {}

fn main() {
    // Due to #145880, this argument isn't an extending context.
    println!("{:?}", { &temp() });
    //~^ ERROR temporary value dropped while borrowed

    // Subexpressions of function call expressions are not extending.
    println!("{:?}{:?}", (), { std::convert::identity(&temp()) });
    //~^ ERROR temporary value dropped while borrowed
}
