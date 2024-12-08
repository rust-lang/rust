//@ run-pass
#![allow(dead_code)]

enum Foo {
    Bar,
    Baz,
    Boo,
}

static X: Foo = Foo::Bar;

pub fn main() {
    match X {
        Foo::Bar => {}
        Foo::Baz | Foo::Boo => panic!()
    }
    match Y {
        Foo::Baz => {}
        Foo::Bar | Foo::Boo => panic!()
    }
}

static Y: Foo = Foo::Baz;
