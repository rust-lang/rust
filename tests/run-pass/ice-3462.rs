#![warn(clippy::all)]
#![allow(clippy::blacklisted_name)]
#![allow(unused)]

enum Foo {
    Bar,
    Baz,
}

fn bar(foo: Foo) {
    macro_rules! baz {
        () => {
            if let Foo::Bar = foo {}
        };
    }

    baz!();
    baz!();
}

fn main() {}
