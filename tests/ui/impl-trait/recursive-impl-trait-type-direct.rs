//@ check-pass

#![allow(unconditional_recursion)]

fn test() -> impl Sized {
    test()
}

fn main() {}
