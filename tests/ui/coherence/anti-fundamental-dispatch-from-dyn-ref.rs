// Regression test for #148727: downstream crates must not be able to implement
// `DispatchFromDyn` for references to local types.

#![feature(dispatch_from_dyn, arbitrary_self_types)]

use std::ops::{DispatchFromDyn, Receiver};

pub struct Ptr<T: ?Sized>(Box<T>, Box<T>);

impl<'a, T: ?Sized, U: ?Sized> DispatchFromDyn<&'a Ptr<U>> for &'a Ptr<T> {}
//~^ ERROR cannot implement `DispatchFromDyn` for the fundamental type

impl<'a, T: ?Sized, U: ?Sized> DispatchFromDyn<&'a mut Ptr<U>> for &'a mut Ptr<T> {}
//~^ ERROR cannot implement `DispatchFromDyn` for the fundamental type

impl<T: ?Sized> Receiver for Ptr<T> {
    type Target = T;
}

pub trait Trait {
    fn method(self: &Ptr<Self>) {}
}

fn main() {}
