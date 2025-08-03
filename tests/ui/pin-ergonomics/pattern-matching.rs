//@ revisions: pin_ergonomics normal
//@ edition:2024
//@[pin_ergonomics] check-pass
#![cfg_attr(pin_ergonomics, feature(pin_ergonomics))]
#![feature(if_let_guard, negative_impls)]
#![allow(incomplete_features)]

use std::pin::Pin;

// This test verifies that a `&pin mut T` can be projected to a pinned
// reference field `&pin mut T.U`.
// FIXME(pin_ergonomics): it is unsound to project `&pin mut T` to
// `&pin mut T.U` when `U: !Unpin` but `T: Unpin`, or when there exists
// `impl Drop for T` that takes a `&mut self` receiver.

struct Foo<T, U> {
    x: T,
    y: U,
}

enum Bar<T, U> {
    Foo(T, U),
    Bar { x: T, y: U },
}

trait IsPinMut {}
trait IsPinConst {}
impl<T: ?Sized> IsPinMut for Pin<&mut T> {}
impl<T: ?Sized> IsPinConst for Pin<&T> {}

fn assert_pin_mut<T: IsPinMut>(_: T) {}
fn assert_pin_const<T: IsPinConst>(_: T) {}

fn foo<T: Unpin, U: Unpin>(foo_mut: Pin<&mut Foo<T, U>>, foo_const: Pin<&Foo<T, U>>) {
    let Foo { x, y } = foo_mut;
    //[normal]~^ ERROR mismatched types
    assert_pin_mut(x);
    assert_pin_mut(y);

    let Foo { x, y } = foo_const;
    //[normal]~^ ERROR mismatched types
    assert_pin_const(x);
    assert_pin_const(y);
}

fn bar<T: Unpin, U: Unpin>(bar_mut: Pin<&mut Bar<T, U>>, bar_const: Pin<&Bar<T, U>>) {
    match bar_mut {
        Bar::Foo(x, y) => {
            //[normal]~^ ERROR mismatched types
            assert_pin_mut(x);
            assert_pin_mut(y);
        }
        _ if let Bar::Bar { x, y } = bar_mut => {
            //[normal]~^ ERROR mismatched types
            assert_pin_mut(x);
            assert_pin_mut(y);
        }
        _ => {}
    }
    match bar_const {
        Bar::Bar { x, y } => {
            //[normal]~^ ERROR mismatched types
            assert_pin_const(x);
            assert_pin_const(y);
        }
        _ if let Bar::Foo(x, y) = bar_const => {
            //[normal]~^ ERROR mismatched types
            assert_pin_const(x);
            assert_pin_const(y);
        }
        _ => {}
    }
}

fn pin_mut_tuple<T, U>(foo_mut: Pin<&mut (Foo<T, U>,)>, foo_const: Pin<&(Foo<T, U>,)>) {
    let (Foo { x, y },) = foo_mut;
    //[normal]~^ ERROR mismatched types
    assert_pin_mut(x);
    assert_pin_mut(y);
    let (Foo { x, y },) = foo_const;
    //[normal]~^ ERROR mismatched types
    assert_pin_const(x);
    assert_pin_const(y);
}

fn pin_mut_array<T, U>(foo_mut: Pin<&mut [Foo<T, U>; 1]>, foo_const: Pin<&[Foo<T, U>; 1]>) {
    let [Foo { x, y }] = foo_mut;
    //[normal]~^ ERROR expected an array or slice, found `Pin<&mut [Foo<T, U>; 1]>`
    assert_pin_mut(x);
    assert_pin_mut(y);
    let [Foo { x, y }] = foo_const;
    //[normal]~^ ERROR expected an array or slice, found `Pin<&[Foo<T, U>; 1]>`
    assert_pin_const(x);
    assert_pin_const(y);
}

fn pin_mut_slice<T, U>(foo_mut: Pin<&mut [Foo<T, U>]>, foo_const: Pin<&[Foo<T, U>]>) {
    if let [Foo { x, y }] = foo_mut {
        //[normal]~^ ERROR expected an array or slice, found `Pin<&mut [Foo<T, U>]>`
        assert_pin_mut(x);
        assert_pin_mut(y);
    }
    if let [Foo { x, y }] = foo_const {
        //[normal]~^ ERROR expected an array or slice, found `Pin<&[Foo<T, U>]>`
        assert_pin_const(x);
        assert_pin_const(y);
    }
}

fn main() {}
