//@ check-pass

#![allow(unconditional_recursion)]

fn test<'a>() -> impl Sized + 'a {
    let _: () = test::<'a>();
}

fn main() {}
