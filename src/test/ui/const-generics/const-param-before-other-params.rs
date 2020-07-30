#![allow(incomplete_features)]
#![feature(const_generics)]

fn bar<const X: (), 'a>(_: &'a ()) {
    //~^ ERROR lifetime parameters must be declared prior to const parameters
}

fn foo<const X: (), T>(_: &T) {}

fn main() {}
