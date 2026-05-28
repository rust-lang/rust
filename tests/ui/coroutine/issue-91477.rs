#![feature(coroutines)]

fn foo() -> impl Sized {
    yield 1; //~ ERROR E0627
    //~^ ERROR: `yield` can only be used in
}

fn main() {}
