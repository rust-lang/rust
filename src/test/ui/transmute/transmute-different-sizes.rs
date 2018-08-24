// normalize-stderr-test "\d+ bits" -> "N bits"

// Tests that `transmute` cannot be called on types of different size.

#![allow(warnings)]
#![feature(specialization)]

use std::mem::transmute;

unsafe fn f() {
    let _: i8 = transmute(16i16);
    //~^ ERROR transmute called with types of different sizes
}

unsafe fn g<T>(x: &T) {
    let _: i8 = transmute(x);
    //~^ ERROR transmute called with types of different sizes
}

trait Specializable { type Output; }

impl<T> Specializable for T {
    default type Output = u16;
}

unsafe fn specializable<T>(x: u16) -> <T as Specializable>::Output {
    transmute(x)
    //~^ ERROR transmute called with types of different sizes
}

fn main() {}
