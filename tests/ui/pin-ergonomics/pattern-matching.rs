//@ revisions: pin_ergonomics normal
//@ edition:2024
//@[pin_ergonomics] check-pass
#![cfg_attr(pin_ergonomics, feature(pin_ergonomics))]
#![feature(if_let_guard)]
#![allow(incomplete_features)]

use std::pin::Pin;

// This test verifies that a `&pin mut Foo` can be projected to a pinned
// reference `&pin mut T` of a `?Unpin` field , and can be projected to
// an unpinned reference `&mut U` of an `Unpin` field.

struct Foo<T, U> {
    x: T,
    y: U,
}

struct Bar<T, U>(T, U);

enum Baz<T, U> {
    Foo(T, U),
    Bar { x: T, y: U },
}

trait IsPinMut {}
trait IsPinConst {}
impl<T: ?Sized> IsPinMut for Pin<&mut T> {}
impl<T: ?Sized> IsPinConst for Pin<&T> {}

fn assert_pin_mut<T: IsPinMut>(_: T) {}
fn assert_pin_const<T: IsPinConst>(_: T) {}

fn foo_mut<T, U: Unpin>(foo: Pin<&mut Foo<T, U>>) {
    let Foo { x, y } = foo;
    //[normal]~^ ERROR mismatched types
    assert_pin_mut(x);
    assert_pin_mut(y);
}

fn foo_const<T, U: Unpin>(foo: Pin<&Foo<T, U>>) {
    let Foo { x, y } = foo;
    //[normal]~^ ERROR mismatched types
    assert_pin_const(x);
    assert_pin_const(y);
}

fn bar_mut<T, U: Unpin>(bar: Pin<&mut Bar<T, U>>) {
    let Bar(x, y) = bar;
    //[normal]~^ ERROR mismatched types
    assert_pin_mut(x);
    assert_pin_mut(y);
}

fn bar_const<T, U: Unpin>(bar: Pin<&Bar<T, U>>) {
    let Bar(x, y) = bar;
    //[normal]~^ ERROR mismatched types
    assert_pin_const(x);
    assert_pin_const(y);
}

fn foo_bar_mut<T, U: Unpin>(foo: Pin<&mut Foo<Bar<T, U>, Bar<T, U>>>) {
    let Foo { x: Bar(x, y), y: Bar(z, w) } = foo;
    //[normal]~^ ERROR mismatched types
    assert_pin_mut(x);
    assert_pin_mut(y);
    assert_pin_mut(z);
    assert_pin_mut(w);
}

fn foo_bar_const<T, U: Unpin>(foo: Pin<&Foo<Bar<T, U>, Bar<T, U>>>) {
    let Foo { x: Bar(x, y), y: Bar(z, w) } = foo;
    //[normal]~^ ERROR mismatched types
    assert_pin_const(x);
    assert_pin_const(y);
    assert_pin_const(z);
    assert_pin_const(w);
}

fn baz_mut<T, U: Unpin>(baz: Pin<&mut Baz<T, U>>) {
    match baz {
        Baz::Foo(x, y) => {
            //[normal]~^ ERROR mismatched types
            assert_pin_mut(x);
            assert_pin_mut(y);
        }
        Baz::Bar { x, y } => {
            //[normal]~^ ERROR mismatched types
            assert_pin_mut(x);
            assert_pin_mut(y);
        }
    }
}

fn baz_const<T, U: Unpin>(baz: Pin<&Baz<T, U>>) {
    match baz {
        Baz::Foo(x, y) => {
            //[normal]~^ ERROR mismatched types
            assert_pin_const(x);
            assert_pin_const(y);
        }
        Baz::Bar { x, y } => {
            //[normal]~^ ERROR mismatched types
            assert_pin_const(x);
            assert_pin_const(y);
        }
    }
}

fn baz_baz_mut<T, U: Unpin>(baz: Pin<&mut Baz<Baz<T, U>, Baz<T, U>>>) {
    match baz {
        Baz::Foo(Baz::Foo(x, y), Baz::Foo(z, w) | Baz::Bar { x: z, y: w }) => {
            //[normal]~^ ERROR mismatched types
            assert_pin_mut(x);
            assert_pin_mut(y);
            assert_pin_mut(z);
            assert_pin_mut(w);
        }
        Baz::Foo(Baz::Bar { x, y }, Baz::Foo(z, w) | Baz::Bar { x: z, y: w }) => {
            //[normal]~^ ERROR mismatched types
            assert_pin_mut(x);
            assert_pin_mut(y);
            assert_pin_mut(z);
            assert_pin_mut(w);
        }
        Baz::Bar { x: Baz::Foo(x, y), y: Baz::Foo(z, w) | Baz::Bar { x: z, y: w } } => {
            //[normal]~^ ERROR mismatched types
            assert_pin_mut(x);
            assert_pin_mut(y);
            assert_pin_mut(z);
            assert_pin_mut(w);
        }
        Baz::Bar { x: Baz::Bar { x, y }, y: Baz::Foo(z, w) | Baz::Bar { x: z, y: w } } => {
            //[normal]~^ ERROR mismatched types
            assert_pin_mut(x);
            assert_pin_mut(y);
            assert_pin_mut(z);
            assert_pin_mut(w);
        }
    }
}

fn baz_baz_const<T, U: Unpin>(baz: Pin<&Baz<Baz<T, U>, Baz<T, U>>>) {
    match baz {
        Baz::Foo(foo, _) if let Baz::Foo(x, y) = foo => {
            //[normal]~^ ERROR mismatched types
            assert_pin_const(x);
            assert_pin_const(y);
        }
        Baz::Bar { x: _, y: bar } if let Baz::Bar { x, y } = bar => {
            //[normal]~^ ERROR mismatched types
            assert_pin_const(x);
            assert_pin_const(y);
        }
        Baz::Foo(Baz::Foo(x, y), Baz::Foo(z, w) | Baz::Bar { x: z, y: w }) => {
            //[normal]~^ ERROR mismatched types
            assert_pin_const(x);
            assert_pin_const(y);
            assert_pin_const(z);
            assert_pin_const(w);
        }
        Baz::Foo(Baz::Bar { x, y }, Baz::Foo(z, w) | Baz::Bar { x: z, y: w }) => {
            //[normal]~^ ERROR mismatched types
            assert_pin_const(x);
            assert_pin_const(y);
            assert_pin_const(z);
            assert_pin_const(w);
        }
        Baz::Bar { x: Baz::Foo(x, y), y: Baz::Foo(z, w) | Baz::Bar { x: z, y: w } } => {
            //[normal]~^ ERROR mismatched types
            assert_pin_const(x);
            assert_pin_const(y);
            assert_pin_const(z);
            assert_pin_const(w);
        }
        Baz::Bar { x: Baz::Bar { x, y }, y: Baz::Foo(z, w) | Baz::Bar { x: z, y: w } } => {
            //[normal]~^ ERROR mismatched types
            assert_pin_const(x);
            assert_pin_const(y);
            assert_pin_const(z);
            assert_pin_const(w);
        }
    }
}

fn main() {}
