// The purpose of this file is to track the error messages from Pin and DerefMut interacting.
//
// Identical to `pin-deref.rs` except for using unstable `const fn`.

//@ check-fail

#![feature(const_convert)]
#![feature(const_trait_impl)]

use std::ops::DerefMut;
use std::pin::Pin;

struct MyUnpinType {}

impl MyUnpinType {
    const fn at_self(&self) {}
    const fn at_mut_self(&mut self) {}
}

struct MyPinType(core::marker::PhantomPinned);

impl MyPinType {
    const fn at_self(&self) {}
    const fn at_mut_self(&mut self) {}
}

const fn call_mut_ref_unpin(mut r_unpin: Pin<&mut MyUnpinType>) {
    r_unpin.at_self();
    r_unpin.at_mut_self();
}

const fn call_ref_unpin(mut r_unpin: Pin<&MyUnpinType>) {
    r_unpin.at_self();
    r_unpin.at_mut_self(); //~ ERROR: cannot borrow data in dereference of `Pin<&MyUnpinType>` as mutable
}

const fn call_mut_ref_pin(mut r_pin: Pin<&mut MyPinType>) {
    r_pin.at_self();
    r_pin.at_mut_self(); //~ ERROR: cannot borrow data in dereference of `Pin<&mut MyPinType>` as mutable
}

const fn call_ref_pin(mut r_pin: Pin<&MyPinType>) {
    r_pin.at_self();
    r_pin.at_mut_self(); //~ ERROR: cannot borrow data in dereference of `Pin<&MyPinType>` as mutable
}

const fn coerce_unpin_rr<'a>(mut r_unpin: &'a mut Pin<&MyUnpinType>) -> &'a MyUnpinType {
    r_unpin
}

const fn coerce_unpin_rm<'a>(mut r_unpin: &'a mut Pin<&MyUnpinType>) -> &'a mut MyUnpinType {
    r_unpin //~ ERROR: cannot borrow data in dereference of `Pin<&MyUnpinType>` as mutable
}

const fn coerce_unpin_mr<'a>(mut r_unpin: &'a mut Pin<&mut MyUnpinType>) -> &'a MyUnpinType {
    r_unpin
}

const fn coerce_unpin_mm<'a>(mut r_unpin: &'a mut Pin<&mut MyUnpinType>) -> &'a mut MyUnpinType {
    r_unpin
}

const fn coerce_pin_rr<'a>(mut r_pin: &'a mut Pin<&MyPinType>) -> &'a MyPinType {
    r_pin
}

const fn coerce_pin_rm<'a>(mut r_pin: &'a mut Pin<&MyPinType>) -> &'a mut MyPinType {
    r_pin //~ ERROR: cannot borrow data in dereference of `Pin<&MyPinType>` as mutable
}

const fn coerce_pin_mr<'a>(mut r_pin: &'a mut Pin<&mut MyPinType>) -> &'a MyPinType {
    r_pin
}

const fn coerce_pin_mm<'a>(mut r_pin: &'a mut Pin<&mut MyPinType>) -> &'a mut MyPinType {
    r_pin //~ ERROR: cannot borrow data in dereference of `Pin<&mut MyPinType>` as mutable
}

fn main() {}
