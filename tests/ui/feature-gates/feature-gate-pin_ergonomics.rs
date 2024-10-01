#![allow(dead_code, incomplete_features)]

use std::pin::Pin;

struct Foo;

impl Foo {
    fn foo(self: Pin<&mut Self>) {
    }
}

fn foo(_: Pin<&mut Foo>) {
}

fn bar(mut x: Pin<&mut Foo>) {
    foo(x);
    foo(x); //~ ERROR use of moved value: `x`
}

fn baz(mut x: Pin<&mut Foo>) {
    x.foo();
    x.foo(); //~ ERROR use of moved value: `x`
}

fn main() {}
