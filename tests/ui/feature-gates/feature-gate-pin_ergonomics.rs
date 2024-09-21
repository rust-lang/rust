#![allow(dead_code, incomplete_features)]

use std::pin::Pin;

struct Foo;

fn foo(_: Pin<&mut Foo>) {
}

fn bar(mut x: Pin<&mut Foo>) {
    foo(x);
    foo(x); //~ ERROR use of moved value: `x`
}

fn main() {}
