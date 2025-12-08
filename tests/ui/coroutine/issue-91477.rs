#![feature(coroutines)]

fn foo() -> impl Sized {
    1.yield; //~ ERROR E0627
    //~^ ERROR: `yield` can only be used in
}

fn main() {}
