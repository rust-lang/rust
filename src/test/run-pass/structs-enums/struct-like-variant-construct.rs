// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

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
