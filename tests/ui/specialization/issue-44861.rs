#![crate_type = "lib"]
#![feature(specialization)]
#![feature(unsize, coerce_unsized)]
#![allow(incomplete_features)]

use std::ops::CoerceUnsized;

pub struct SmartassPtr<A: Smartass+?Sized>(A::Data);

pub trait Smartass {
    type Data;
    type Data2: CoerceUnsized<*const [u8]>;
}

pub trait MaybeDynCompatible {}

impl MaybeDynCompatible for () {}

impl<T> Smartass for T {
    type Data = <Self as Smartass>::Data2;
    default type Data2 = ();
    //~^ ERROR: the trait bound `(): CoerceUnsized<*const [u8]>` is not satisfied
}

impl Smartass for () {
    type Data2 = *const [u8; 1];
}

impl Smartass for dyn MaybeDynCompatible {
    type Data = *const [u8];
    type Data2 = *const [u8; 0];
}

impl<U: Smartass+?Sized, T: Smartass+?Sized> CoerceUnsized<SmartassPtr<T>> for SmartassPtr<U>
    where <U as Smartass>::Data: std::ops::CoerceUnsized<<T as Smartass>::Data>
{}

pub fn conv(s: SmartassPtr<()>) -> SmartassPtr<dyn MaybeDynCompatible> {
    s
}
