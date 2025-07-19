//@ revisions: pin_ergonomics normal
//@ edition:2024
//@[pin_ergonomics] check-pass
#![cfg_attr(pin_ergonomics, feature(pin_ergonomics))]
#![feature(if_let_guard, negative_impls)]
#![allow(incomplete_features)]

use std::pin::Pin;

// This test verifies that a `&pin mut T` can be projected to a pinned
// reference field `&pin mut T.U` when `T: !Unpin` or when `U: Unpin`.

struct Foo<T, U> {
    x: T,
    y: U,
}

struct NegUnpinFoo<T, U> {
    x: T,
    y: U,
}

enum NegUnpinBar<T, U> {
    Foo(T, U),
    Bar { x: T, y: U },
}

impl<T, U> !Unpin for NegUnpinFoo<T, U> {}
impl<T, U> !Unpin for NegUnpinBar<T, U> {}

trait IsPinMut {}
trait IsPinConst {}
impl<T: ?Sized> IsPinMut for Pin<&mut T> {}
impl<T: ?Sized> IsPinConst for Pin<&T> {}

fn assert_pin_mut<T: IsPinMut>(_: T) {}
fn assert_pin_const<T: IsPinConst>(_: T) {}

// Pinned references can be projected to pinned references of `Unpin` fields
fn unpin2unpin<T: Unpin, U: Unpin>(foo_mut: Pin<&mut Foo<T, U>>, foo_const: Pin<&Foo<T, U>>) {
    let Foo { x, y } = foo_mut;
    //[normal]~^ ERROR mismatched types
    assert_pin_mut(x);
    assert_pin_mut(y);

    let Foo { x, y } = foo_const;
    //[normal]~^ ERROR mismatched types
    assert_pin_const(x);
    assert_pin_const(y);
}

// Pinned references can be only projected to pinned references of `Unpin` fields
fn unpin2partial_unpin<T: Unpin, U>(foo_mut: Pin<&mut Foo<T, U>>, foo_const: Pin<&Foo<T, U>>) {
    let Foo { x, .. } = foo_mut;
    //[normal]~^ ERROR mismatched types
    assert_pin_mut(x);

    let Foo { x, .. } = foo_const;
    //[normal]~^ ERROR mismatched types
    assert_pin_const(x);
}

// Pinned references of `!Unpin` types can be projected to pinned references
fn neg_unpin2not_unpin<T, U>(
    foo_mut: Pin<&mut NegUnpinFoo<T, U>>,
    foo_const: Pin<&NegUnpinFoo<T, U>>,
) {
    let NegUnpinFoo { x, y } = foo_mut;
    //[normal]~^ ERROR mismatched types
    assert_pin_mut(x);
    assert_pin_mut(y);

    let NegUnpinFoo { x, y } = foo_const;
    //[normal]~^ ERROR mismatched types
    assert_pin_const(x);
    assert_pin_const(y);
}

// Pinned references of `!Unpin` types can be projected to pinned references of `Unpin` fields
fn neg_unpin2unpin<T: Unpin, U: Unpin>(
    bar_mut: Pin<&mut NegUnpinBar<T, U>>,
    bar_const: Pin<&NegUnpinBar<T, U>>,
) {
    match bar_mut {
        NegUnpinBar::Foo(x, y) => {
            //[normal]~^ ERROR mismatched types
            assert_pin_mut(x);
            assert_pin_mut(y);
        }
        _ if let NegUnpinBar::Bar { x, y } = bar_mut => {
            //[normal]~^ ERROR mismatched types
            assert_pin_mut(x);
            assert_pin_mut(y);
        }
        _ => {}
    }
    match bar_const {
        NegUnpinBar::Bar { x, y } => {
            //[normal]~^ ERROR mismatched types
            assert_pin_const(x);
            assert_pin_const(y);
        }
        _ if let NegUnpinBar::Foo(x, y) = bar_const => {
            //[normal]~^ ERROR mismatched types
            assert_pin_const(x);
            assert_pin_const(y);
        }
        _ => {}
    }
}

fn neg_unpin_tuple<T, U>(
    foo_mut: Pin<&mut (NegUnpinFoo<T, U>,)>,
    foo_const: Pin<&(NegUnpinFoo<T, U>,)>,
) {
    let (NegUnpinFoo { x, y },) = foo_mut;
    //[normal]~^ ERROR mismatched types
    assert_pin_mut(x);
    assert_pin_mut(y);
    let (NegUnpinFoo { x, y },) = foo_const;
    //[normal]~^ ERROR mismatched types
    assert_pin_const(x);
    assert_pin_const(y);
}

fn neg_unpin_array<T, U>(
    foo_mut: Pin<&mut [NegUnpinFoo<T, U>; 1]>,
    foo_const: Pin<&[NegUnpinFoo<T, U>; 1]>,
) {
    let [NegUnpinFoo { x, y }] = foo_mut;
    //[normal]~^ ERROR expected an array or slice, found `Pin<&mut [NegUnpinFoo<T, U>; 1]>`
    assert_pin_mut(x);
    assert_pin_mut(y);
    let [NegUnpinFoo { x, y }] = foo_const;
    //[normal]~^ ERROR expected an array or slice, found `Pin<&[NegUnpinFoo<T, U>; 1]>`
    assert_pin_const(x);
    assert_pin_const(y);
}

fn neg_unpin_slice<T, U>(
    foo_mut: Pin<&mut [NegUnpinFoo<T, U>]>,
    foo_const: Pin<&[NegUnpinFoo<T, U>]>,
) {
    if let [NegUnpinFoo { x, y }] = foo_mut {
        //[normal]~^ ERROR expected an array or slice, found `Pin<&mut [NegUnpinFoo<T, U>]>`
        assert_pin_mut(x);
        assert_pin_mut(y);
    }
    if let [NegUnpinFoo { x, y }] = foo_const {
        //[normal]~^ ERROR expected an array or slice, found `Pin<&[NegUnpinFoo<T, U>]>`
        assert_pin_const(x);
        assert_pin_const(y);
    }
}

fn main() {}
