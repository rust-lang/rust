//@ known-bug: #74451
//@ compile-flags: -Copt-level=0

#![feature(specialization)]
#![feature(unsize, coerce_unsized)]
#![allow(incomplete_features)]
#![crate_type = "lib"]

use std::ops::CoerceUnsized;

pub struct SmartassPtr<A: Smartass+?Sized>(A::Data);

pub trait Smartass {
    type Data;
    type Data2: CoerceUnsized<*const [u8]>;
}

pub trait MaybeObjectSafe {}

impl MaybeObjectSafe for () {}

impl<T> Smartass for T {
    type Data = <Self as Smartass>::Data2;
    default type Data2 = *const [u8; 0];
}

impl Smartass for () {
    type Data2 = *const [u8; 1];
}

impl Smartass for dyn MaybeObjectSafe {
    type Data = *const [u8];
    type Data2 = *const [u8; 0];
}

impl<U: Smartass+?Sized, T: Smartass+?Sized> CoerceUnsized<SmartassPtr<T>> for SmartassPtr<U>
    where <U as Smartass>::Data: std::ops::CoerceUnsized<<T as Smartass>::Data>
{}

pub fn conv(s: SmartassPtr<()>) -> SmartassPtr<dyn MaybeObjectSafe> {
    s // This shouldn't coerce
}
