#![feature(type_ascription)]

enum Bug<S> { //~ ERROR parameter `S` is never used
    Var = 0: S,
    //~^ ERROR generic parameters may not be used
}

fn main() {}
