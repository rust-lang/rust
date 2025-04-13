//@ revisions: pin_ergonomics normal
//@ edition:2024
#![cfg_attr(pin_ergonomics, feature(pin_ergonomics))]
#![feature(deref_patterns)]
#![allow(incomplete_features)]

// This test verifies that the `pin_ergonomics` feature works well
// together with the `deref_patterns` feature under the error:
// "mix of deref patterns and normal constructors".

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
    match foo {
        Foo { .. } => {} //~ ERROR mix of deref patterns and normal constructors
        Pin { .. } => {}
    }
    let _ = || match foo {
        Foo { .. } => {} //~ ERROR mix of deref patterns and normal constructors
        Pin { .. } => {}
    };

    #[cfg(pin_ergonomics)]
    match foo {
        Foo { x, y } => {} //[pin_ergonomics]~ ERROR mix of deref patterns and normal constructors
        Pin { .. } => {}
    }
    #[cfg(pin_ergonomics)]
    let _ = || match foo {
        Foo { .. } => {} //[pin_ergonomics]~ ERROR mix of deref patterns and normal constructors
        Pin { .. } => {}
    };
}

fn foo_const<T: Unpin, U: Unpin>(foo: Pin<&Foo<T, U>>) {
    match foo {
        Foo { .. } => {} //~ ERROR mix of deref patterns and normal constructors
        Pin { .. } => {}
    }
    let _ = || match foo {
        Foo { .. } => {} //~ ERROR mix of deref patterns and normal constructors
        Pin { .. } => {}
    };

    #[cfg(pin_ergonomics)]
    match foo {
        Foo { x, y } => {} //[pin_ergonomics]~ ERROR mix of deref patterns and normal constructors
        Pin { .. } => {}
    }
    #[cfg(pin_ergonomics)]
    let _ = || match foo {
        Foo { .. } => {} //[pin_ergonomics]~ ERROR mix of deref patterns and normal constructors
        Pin { .. } => {}
    };
}

fn bar_mut<T: Unpin, U: Unpin>(bar: Pin<&mut Bar<T, U>>) {
    match bar {
        Bar(..) => {} //~ ERROR mix of deref patterns and normal constructors
        Pin { .. } => {}
    }
    let _ = || match bar {
        Bar(..) => {} //~ ERROR mix of deref patterns and normal constructors
        Pin { .. } => {}
    };

    #[cfg(pin_ergonomics)]
    match bar {
        Bar(x, y) => {} //[pin_ergonomics]~ ERROR mix of deref patterns and normal constructors
        Pin { .. } => {}
    }
    #[cfg(pin_ergonomics)]
    let _ = || match bar {
        Bar(x, y) => {} //[pin_ergonomics]~ ERROR mix of deref patterns and normal constructors
        Pin { .. } => {}
    };
}

fn bar_const<T: Unpin, U: Unpin>(bar: Pin<&Bar<T, U>>) {
    match bar {
        Bar(..) => {} //~ ERROR mix of deref patterns and normal constructors
        Pin { .. } => {}
    }
    let _ = || match bar {
        Bar(..) => {} //~ ERROR mix of deref patterns and normal constructors
        Pin { .. } => {}
    };

    #[cfg(pin_ergonomics)]
    match bar {
        Bar(x, y) => {} //[pin_ergonomics]~ ERROR mix of deref patterns and normal constructors
        Pin { .. } => {}
    }
    #[cfg(pin_ergonomics)]
    let _ = || match bar {
        Bar(x, y) => {} //[pin_ergonomics]~ ERROR mix of deref patterns and normal constructors
        Pin { .. } => {}
    };
}

fn main() {}
