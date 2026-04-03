// Regression test for #52560
// Verify that deriving Debug on a struct with associated types
// gives a helpful diagnostic suggesting manual Debug implementation
// instead of incorrectly suggesting adding a Debug bound on the
// type parameter that isn't directly used as a field.

use std::fmt::Debug;

#[derive(Debug)]
struct Foo<B: Bar>(B::Item);

trait Bar {
    type Item: Debug;
}

fn foo<B: Bar>(f: Foo<B>) {
    println!("{:?}", f);
    //~^ ERROR `B` doesn't implement `Debug`
}

fn main() {}
