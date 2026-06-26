//@ revisions: pin_ergonomics normal
//@ edition:2024
//@[pin_ergonomics] check-pass
#![cfg_attr(pin_ergonomics, feature(pin_ergonomics))]
#![feature(deref_patterns)]
#![allow(incomplete_features)]

// This test verifies that the `pin_ergonomics` feature works well
// together with the `deref_patterns` feature.

use std::pin::Pin;

#[cfg_attr(pin_ergonomics, pin_v2)]
struct Foo<T, U> {
    x: T,
    y: U,
}

#[cfg_attr(pin_ergonomics, pin_v2)]
struct Bar<T, U>(T, U);

#[cfg_attr(pin_ergonomics, pin_v2)]
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

fn foo_mut<T: Unpin, U: Unpin>(mut foo: Pin<&mut Foo<T, U>>) {
    let Foo { .. } = foo.as_mut();
    let Foo { x, y } = foo.as_mut();
    //[normal]~^ ERROR cannot move out of a shared reference
    #[cfg(pin_ergonomics)]
    assert_pin_mut(x);
    #[cfg(pin_ergonomics)]
    assert_pin_mut(y);
    let Pin { .. } = foo.as_mut();

    let _ = || {
        let Foo { .. } = foo.as_mut();
        let Foo { x, y } = foo.as_mut();
        //[normal]~^ ERROR cannot move out of a shared reference
        #[cfg(pin_ergonomics)]
        assert_pin_mut(x);
        #[cfg(pin_ergonomics)]
        assert_pin_mut(y);
        let Pin { .. } = foo.as_mut();
    };
}

fn foo_const<T: Unpin, U: Unpin>(foo: Pin<&Foo<T, U>>) {
    let Foo { .. } = foo.as_ref();
    let Foo { x, y } = foo.as_ref();
    //[normal]~^ ERROR cannot move out of a shared reference
    #[cfg(pin_ergonomics)]
    assert_pin_const(x);
    #[cfg(pin_ergonomics)]
    assert_pin_const(y);
    let Pin { .. } = foo.as_ref();

    let _ = || {
        let Foo { .. } = foo.as_ref();
        let Foo { x, y } = foo.as_ref();
        //[normal]~^ ERROR cannot move out of a shared reference
        #[cfg(pin_ergonomics)]
        assert_pin_const(x);
        #[cfg(pin_ergonomics)]
        assert_pin_const(y);
        let Pin { .. } = foo.as_ref();
    };
}

fn bar_mut<T: Unpin, U: Unpin>(mut bar: Pin<&mut Bar<T, U>>) {
    let Bar(..) = bar.as_mut();
    let Bar(x, y) = bar.as_mut();
    //[normal]~^ ERROR cannot move out of a shared reference
    #[cfg(pin_ergonomics)]
    assert_pin_mut(x);
    #[cfg(pin_ergonomics)]
    assert_pin_mut(y);
    let Pin { .. } = bar.as_mut();

    let _ = || {
        let Bar(..) = bar.as_mut();
        let Bar(x, y) = bar.as_mut();
        //[normal]~^ ERROR cannot move out of a shared reference
        #[cfg(pin_ergonomics)]
        assert_pin_mut(x);
        #[cfg(pin_ergonomics)]
        assert_pin_mut(y);
        let Pin { .. } = bar.as_mut();
    };
}

fn bar_const<T: Unpin, U: Unpin>(bar: Pin<&Bar<T, U>>) {
    let Bar(..) = bar.as_ref();
    let Bar(x, y) = bar.as_ref();
    //[normal]~^ ERROR cannot move out of a shared reference
    #[cfg(pin_ergonomics)]
    assert_pin_const(x);
    #[cfg(pin_ergonomics)]
    assert_pin_const(y);
    let Pin { .. } = bar.as_ref();

    let _ = || {
        let Bar(..) = bar.as_ref();
        let Bar(x, y) = bar.as_ref();
        //[normal]~^ ERROR cannot move out of a shared reference
        #[cfg(pin_ergonomics)]
        assert_pin_const(x);
        #[cfg(pin_ergonomics)]
        assert_pin_const(y);
        let Pin { .. } = bar.as_ref();
    };
}

fn main() {}
