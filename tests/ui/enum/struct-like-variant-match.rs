//@ run-pass
#![allow(non_shorthand_field_patterns)]

enum Foo {
    Bar {
        x: isize,
        y: isize
    },
    Baz {
        x: f64,
        y: f64
    }
}

fn f(x: &Foo) {
    match *x {
        Foo::Baz { x: x, y: y } => {
            assert_eq!(x, 1.0);
            assert_eq!(y, 2.0);
        }
        Foo::Bar { y: y, x: x } => {
            assert_eq!(x, 1);
            assert_eq!(y, 2);
        }
    }
}

pub fn main() {
    let x = Foo::Bar { x: 1, y: 2 };
    f(&x);
    let y = Foo::Baz { x: 1.0, y: 2.0 };
    f(&y);
}
