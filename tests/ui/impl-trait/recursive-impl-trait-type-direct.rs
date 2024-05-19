#![allow(unconditional_recursion)]

fn test() -> impl Sized {
    //~^ ERROR undefined opaque type
    test()
}

fn main() {}
