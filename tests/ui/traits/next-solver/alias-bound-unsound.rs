//@ compile-flags: -Znext-solver

// Makes sure that alias bounds are not unsound!

#![feature(trivial_bounds)]

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
    //~^ ERROR impl has stricter requirements than trait
}

fn main() {
    let x = String::from("hello, world");
    drop(<() as Foo>::copy_me(&x));
    //~^ ERROR type mismatch resolving `<() as Foo>::Item normalizes-to String`
    //~| ERROR mismatched types
    //~| ERROR type mismatch resolving `<() as Foo>::Item normalizes-to String`
    println!("{x}");
}
