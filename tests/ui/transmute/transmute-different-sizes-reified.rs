//@ normalize-stderr: "\d+ bits" -> "N bits"

// Tests that `transmute` cannot be indirectly called on types of different size.

#![allow(warnings)]
#![feature(specialization)]

use std::mem::transmute;

unsafe fn f() {
    let _: i8 = (transmute as unsafe fn(_) -> _)(16i16);
    //~^ ERROR cannot transmute between types of different sizes, or dependently-sized types
}

unsafe fn g<T>(x: &T) {
    let _: i8 = (transmute as unsafe fn(_) -> _)(x);
    //~^ ERROR cannot transmute between types of different sizes, or dependently-sized types
}

trait Specializable { type Output; }

impl<T> Specializable for T {
    default type Output = u16;
}

unsafe fn specializable<T>(x: u16) -> <T as Specializable>::Output {
    (transmute as unsafe fn(_) -> _)(x)
    //~^ ERROR cannot transmute between types of different sizes, or dependently-sized types
}

fn main() {}
