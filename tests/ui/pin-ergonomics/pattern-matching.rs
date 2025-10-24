//@ revisions: pin_ergonomics normal
//@ edition:2024
#![cfg_attr(pin_ergonomics, feature(pin_ergonomics))]
#![feature(if_let_guard, negative_impls)]
#![allow(incomplete_features)]

use std::pin::Pin;

// This test verifies that a `&pin mut T` can be projected to a pinned
// reference field `&pin mut T.U` when `T` is marked with `#[pin_v2]`.

#[pin_v2] //[normal]~ ERROR the `#[pin_v2]` attribute is an experimental feature
struct Foo<T, U> {
    x: T,
    y: U,
}

#[pin_v2] //[normal]~ ERROR the `#[pin_v2]` attribute is an experimental feature
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

fn tuple_ref_mut_pat_and_pin_mut_of_tuple_mut_ty<'a, T, U>(
    (&mut x,): Pin<&'a mut (&'a mut Foo<T, U>,)>, //[normal]~ ERROR mismatched type
) -> Pin<&'a mut Foo<T, U>> {
    x
}

fn tuple_ref_mut_pat_and_pin_mut_of_mut_tuple_ty<'a, T, U>(
    (&mut x,): Pin<&'a mut &'a mut (Foo<T, U>,)>, //~ ERROR mismatched type
) -> Pin<&'a mut Foo<T, U>> {
    x
}

fn ref_mut_tuple_pat_and_pin_mut_of_tuple_mut_ty<'a, T, U>(
    &mut (x,): Pin<&'a mut (&'a mut Foo<T, U>,)>, //~ ERROR mismatched type
) -> Pin<&'a mut Foo<T, U>> {
    x
}

fn ref_mut_tuple_pat_and_pin_mut_of_mut_tuple_ty<'a, T, U>(
    &mut (x,): Pin<&'a mut &'a mut (Foo<T, U>,)>, //~ ERROR mismatched type
) -> Pin<&'a mut Foo<T, U>> {
    x
}

fn tuple_pat_and_pin_mut_of_tuple_mut_ty<'a, T, U>(
    (x,): Pin<&'a mut (&'a mut Foo<T, U>,)>, //[normal]~ ERROR mismatched type
) -> Pin<&'a mut &'a mut Foo<T, U>> {
    x // ok
}

fn tuple_pat_and_pin_mut_of_mut_tuple_ty<'a, T, U>(
    (x,): Pin<&'a mut &'a mut (Foo<T, U>,)>, //[normal]~ ERROR mismatched type
) -> Pin<&'a mut Foo<T, U>> {
    x
}

fn main() {}
