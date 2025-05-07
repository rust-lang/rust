//@ run-pass
#![allow(dead_code)]

enum Foo {
    Bar {
        a: isize,
        b: isize
    },
    Baz {
        c: f64,
        d: f64
    }
}

pub fn main() {
    let _x = Foo::Bar { a: 2, b: 3 };
}
