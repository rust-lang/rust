#![feature(pin_ergonomics)]
#![allow(dead_code, incomplete_features)]

// make sure we can't accidentally reborrow Pin<&T> as Pin<&mut T>

use std::pin::Pin;

struct Foo;

fn foo(_: Pin<&mut Foo>) {
}

fn bar(x: Pin<&Foo>) {
    foo(x); //~ ERROR mismatched types
            //| ERROR types differ in mutability
}

fn main() {}
