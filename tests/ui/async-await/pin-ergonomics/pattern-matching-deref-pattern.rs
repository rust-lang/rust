//@ revisions: pin_ergonomics normal
//@ edition:2024
//@ check-pass
// //@[normal] check-pass
// //@[pin_ergonomics] rustc-env:RUSTC_LOG=rustc_hir_typeck::expr_use_visitor=DEBUG
#![cfg_attr(pin_ergonomics, feature(pin_ergonomics))]
#![feature(deref_patterns)]
#![allow(incomplete_features)]

// This test verifies that the `pin_ergonomics` feature works well
// together with the `deref_patterns` feature.

use std::pin::Pin;

struct Foo<T, U> {
    x: T,
    y: U,
}

struct Bar<T, U>(T, U);

enum Baz<T, U> {
    Foo(T, U),
    Bar { x: T, y: U },
}

fn foo_mut<T: Unpin, U: Unpin>(foo: Pin<&mut Foo<T, U>>) {
    let Foo { .. } = foo;
    let Pin { .. } = foo;
    let _ = || {
        let Foo { .. } = foo;
        let Pin { .. } = foo;
    };

    #[cfg(pin_ergonomics)]
    let Foo { x, y } = foo;
    #[cfg(pin_ergonomics)]
    let _ = || {
        let Foo { x, y } = foo;
    };
}

fn foo_const<T: Unpin, U: Unpin>(foo: Pin<&Foo<T, U>>) {
    let Foo { .. } = foo;
    let Pin { .. } = foo;
    let _ = || {
        let Foo { .. } = foo;
        let Pin { .. } = foo;
    };

    #[cfg(pin_ergonomics)]
    let Foo { x, y } = foo;
    #[cfg(pin_ergonomics)]
    let _ = || {
        let Foo { x, y } = foo;
    };
}

fn bar_mut<T: Unpin, U: Unpin>(bar: Pin<&mut Bar<T, U>>) {
    let Bar(..) = bar;
    let Pin { .. } = bar;
    let _ = || {
        let Bar(..) = bar;
        let Pin { .. } = bar;
    };

    #[cfg(pin_ergonomics)]
    let Bar(x, y) = bar;
    #[cfg(pin_ergonomics)]
    let _ = || {
        let Bar(x, y) = bar;
    };
}

fn bar_const<T: Unpin, U: Unpin>(bar: Pin<&Bar<T, U>>) {
    let Bar(..) = bar;
    let Pin { .. } = bar;
    let _ = || {
        let Bar(..) = bar;
        let Pin { .. } = bar;
    };

    #[cfg(pin_ergonomics)]
    let Bar(x, y) = bar;
    #[cfg(pin_ergonomics)]
    let _ = || {
        let Bar(x, y) = bar;
    };
}

fn main() {}
