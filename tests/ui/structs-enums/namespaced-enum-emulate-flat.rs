//@ run-pass
#![allow(dead_code)]

pub use Foo::*;
use nest::{Bar, D, E, F};

pub enum Foo {
    A,
    B(isize),
    C { a: isize },
}

impl Foo {
    pub fn foo() {}
}

fn _f(f: Foo) {
    match f {
        A | B(_) | C { .. } => {}
    }
}

mod nest {
    pub use self::Bar::*;

    pub enum Bar {
        D,
        E(isize),
        F { a: isize },
    }

    impl Bar {
        pub fn foo() {}
    }
}

fn _f2(f: Bar) {
    match f {
        D | E(_) | F { .. } => {}
    }
}

fn main() {}
