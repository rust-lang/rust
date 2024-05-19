#![warn(clippy::all)]
#![allow(clippy::disallowed_names, clippy::equatable_if_let, clippy::needless_if)]
#![allow(unused)]

/// Test for https://github.com/rust-lang/rust-clippy/issues/3462

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
