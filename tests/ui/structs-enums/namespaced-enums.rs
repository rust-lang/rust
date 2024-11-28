//@ run-pass
#![allow(dead_code)]

enum Foo {
    A,
    B(isize),
    C { a: isize },
}

fn _foo (f: Foo) {
    match f {
        Foo::A | Foo::B(_) | Foo::C { .. } => {}
    }
}

pub fn main() {}
