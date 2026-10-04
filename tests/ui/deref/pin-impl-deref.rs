// The purpose of this file is to track the error messages from Pin and DerefMut interacting.

//@ check-fail

use std::ops::DerefMut;
use std::pin::Pin;

struct MyUnpinType {}

impl MyUnpinType {
    fn at_self(&self) {}
    fn at_mut_self(&mut self) {}
}

struct MyPinType(core::marker::PhantomPinned);

impl MyPinType {
    fn at_self(&self) {}
    fn at_mut_self(&mut self) {}
}

fn impl_deref_mut(_: impl DerefMut) {}
fn unpin_impl_ref(r_unpin: Pin<&MyUnpinType>) {
    impl_deref_mut(r_unpin)
    //~^ ERROR: the trait bound `Pin<&MyUnpinType>: DerefMut` is not satisfied
}
fn unpin_impl_mut(r_unpin: Pin<&mut MyUnpinType>) {
    impl_deref_mut(r_unpin)
}
fn pin_impl_ref(r_pin: Pin<&MyPinType>) {
    impl_deref_mut(r_pin)
    //~^ ERROR: `PhantomPinned` cannot be unpinned
    //~| ERROR: the trait bound `Pin<&MyPinType>: DerefMut` is not satisfied
}
fn pin_impl_mut(r_pin: Pin<&mut MyPinType>) {
    impl_deref_mut(r_pin)
    //~^ ERROR: `PhantomPinned` cannot be unpinned
}

fn main() {}
