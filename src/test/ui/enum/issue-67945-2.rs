#![feature(type_ascription)]

enum Bug<S> {
    Var = 0: S,
    //~^ ERROR: mismatched types
    //~| ERROR: mismatched types
}

fn main() {}
