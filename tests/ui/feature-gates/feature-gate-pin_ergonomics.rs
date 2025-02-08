#![allow(dead_code)]

use std::pin::Pin;

struct Foo;

impl Foo {
    fn foo(self: Pin<&mut Self>) {
    }
}

fn foo(x: Pin<&mut Foo>) {
    let _y: &pin mut Foo = x; //~ ERROR pinned reference syntax is experimental
}

fn foo_sugar(_: &pin mut Foo) {} //~ ERROR pinned reference syntax is experimental

fn bar(x: Pin<&mut Foo>) {
    foo(x);
    foo(x); //~ ERROR use of moved value: `x`
}

fn baz(mut x: Pin<&mut Foo>) {
    x.foo();
    x.foo(); //~ ERROR use of moved value: `x`
}

fn baz_sugar(_: &pin const Foo) {} //~ ERROR pinned reference syntax is experimental

fn pinned_locals() {
    let pin mut x = Foo; //~ ERROR pinned reference syntax is experimental
    let pin const y = Foo; //~ ERROR pinned reference syntax is experimental
    x = Foo; // FIXME: this should be an error
    y = Foo; //~ ERROR cannot assign twice to immutable variable `y`
}

fn main() {}
