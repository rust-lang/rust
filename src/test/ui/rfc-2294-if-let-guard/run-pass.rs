// run-pass

#![feature(if_let_guard)]
#![allow(incomplete_features)]

enum Foo {
    Bar,
    Baz,
    Qux(u8),
}

fn bar(x: bool) -> Foo {
    if x { Foo::Baz } else { Foo::Bar }
}

fn baz(x: u8) -> Foo {
    if x % 2 == 0 { Foo::Bar } else { Foo::Baz }
}

fn qux(x: u8) -> Foo {
    Foo::Qux(x.rotate_left(1))
}

fn main() {
    match Some((true, 3)) {
        Some((x, _)) if let Foo::Bar = bar(x) => panic!(),
        Some((_, x)) if let Foo::Baz = baz(x) => {},
        _ => panic!(),
    }
    match Some(42) {
        Some(x) if let Foo::Qux(y) = qux(x) => assert_eq!(y, 84),
        _ => panic!(),
    }
}
