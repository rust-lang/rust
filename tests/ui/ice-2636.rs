#![allow(dead_code)]

enum Foo {
    A,
    B,
    C,
}

macro_rules! test_hash {
    ($foo:expr, $($t:ident => $ord:expr),+ ) => {
        use self::Foo::*;
        match $foo {
            $ ( & $t => $ord,
            )*
        };
    };
}

fn main() {
    let a = Foo::A;
    test_hash!(&a, A => 0, B => 1, C => 2);
}
