// run-pass

#![feature(if_let_guard)]

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

    // issue #88015
    #[allow(irrefutable_let_patterns)]
    match () {
        () | () if let x = 42 => assert_eq!(x, 42),
        _ => panic!()
    }
}
