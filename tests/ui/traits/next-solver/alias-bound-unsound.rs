//@ compile-flags: -Znext-solver

// Makes sure that alias bounds are not unsound!

#![feature(trivial_bounds)]

// we use identity instead of drop because the presence of [const] Destruct means that there
// are additional bounds on the function, which result in additional errors
use std::convert::identity;

trait Foo {
    type Item: Copy
    where
        <Self as Foo>::Item: Copy;

    fn copy_me(x: &Self::Item) -> Self::Item {
        *x
    }
}

impl Foo for () {
    type Item = String where String: Copy;
    //~^ ERROR overflow evaluating the requirement `String: Copy`
}

fn main() {
    let x = String::from("hello, world");
    let _ = identity(<() as Foo>::copy_me(&x));
    //~^ ERROR type mismatch resolving `<() as Foo>::Item normalizes-to String`
    //~| ERROR type mismatch resolving `<() as Foo>::Item normalizes-to String`
    //~| ERROR mismatched types
    println!("{x}");
}
