#![feature(pin_ergonomics)]
//@ normalize-stderr: "\n\n\z" -> "\n"

use std::marker::PhantomPinned;
use std::pin::Pin;

struct UnpinAdt;

struct NotUnpin {
    _pin: PhantomPinned,
}

#[pin_v2]
struct PinV2NotUnpin {
    _pin: PhantomPinned,
}

struct GenericAdt<T> {
    x: T,
}

impl<T> Unpin for GenericAdt<T> {}

struct DropGenericAdt<T>(T);

impl<T> Drop for DropGenericAdt<T> {
    fn drop(&mut self) {}
}

fn direct_pin_mut_unpin() {
    let _: Pin<&mut _> = &pin mut UnpinAdt;
}

fn direct_pin_const_unpin() {
    let _: Pin<&_> = &pin const UnpinAdt;
}

fn direct_pin_mut_not_unpin() {
    let _ = &pin mut NotUnpin { _pin: PhantomPinned };
    //~^ ERROR cannot directly pin a type that is not structurally pinnable
}

fn direct_pin_const_not_unpin() {
    let _ = &pin const NotUnpin { _pin: PhantomPinned };
    //~^ ERROR cannot directly pin a type that is not structurally pinnable
}

fn direct_pin_mut_pin_v2_not_unpin() {
    let _: Pin<&mut _> = &pin mut PinV2NotUnpin { _pin: PhantomPinned };
}

fn direct_pin_const_pin_v2_not_unpin() {
    let _: Pin<&_> = &pin const PinV2NotUnpin { _pin: PhantomPinned };
}

fn direct_pin_mut_generic<T>(mut value: GenericAdt<T>) {
    let _ = &pin mut value;
    //~^ ERROR cannot directly pin a type that is not structurally pinnable
}

fn direct_pin_const_generic<T>(value: GenericAdt<T>) {
    let _ = &pin const value;
    //~^ ERROR cannot directly pin a type that is not structurally pinnable
}

pub fn unsound_pin_borrow<T>(input: &pin mut GenericAdt<T>) -> &pin mut T {
    &pin mut input.x
    //~^ ERROR cannot directly pin a type that is not structurally pinnable
}

fn direct_pin_mut_manual_unpin_generic() {
    let mut value = GenericAdt { x: NotUnpin { _pin: PhantomPinned } };
    let _ = &pin mut value;
    //~^ ERROR cannot directly pin a type that is not structurally pinnable
}

fn direct_pin_mut_drop_generic(mut input: DropGenericAdt<NotUnpin>) {
    let _: &pin mut _ = &pin mut input;
    //~^ ERROR cannot directly pin a type that is not structurally pinnable
}

fn direct_pin_mut_unpin_then_mut_borrow_and_move() {
    let mut value = UnpinAdt;
    {
        let _ = &pin mut value;
    }
    let _ = &mut value;
    let _ = value;
}

fn direct_pin_const_unpin_then_move() {
    let value = UnpinAdt;
    {
        let _ = &pin const value;
    }
    let _ = value;
}

fn direct_pin_mut_generic_unpin_bound_then_move<T: Unpin>(mut value: T) {
    {
        let _ = &pin mut value;
    }
    let _ = value;
}

fn main() {}
