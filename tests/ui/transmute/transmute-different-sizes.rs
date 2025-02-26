//@ normalize-stderr: "\d+ bits" -> "N bits"

// Tests that `transmute` cannot be called on types of different size.

#![allow(warnings)]
#![feature(specialization)]

use std::mem::transmute;

unsafe fn f() {
    let _: i8 = transmute(16i16);
    //~^ ERROR cannot transmute between types of different sizes, or dependently-sized types
}

unsafe fn g<T>(x: &T) {
    let _: i8 = transmute(x);
    //~^ ERROR cannot transmute between types of different sizes, or dependently-sized types
}

trait Specializable { type Output; }

impl<T> Specializable for T {
    default type Output = u16;
}

unsafe fn specializable<T>(x: u16) -> <T as Specializable>::Output {
    transmute(x)
    //~^ ERROR cannot transmute between types of different sizes, or dependently-sized types
}

#[repr(align(32))]
struct OverAlignZST;
pub struct PtrAndOverAlignZST<T: ?Sized> {
    _inner: *mut T,
    _other: OverAlignZST,
}
pub unsafe fn shouldnt_work<T: ?Sized>(from: *mut T) -> PtrAndOverAlignZST<T> {
    transmute(from)
    //~^ ERROR cannot transmute between types of different sizes, or dependently-sized types
}

pub struct PtrAndEmptyArray<T: ?Sized> {
    _inner: *mut T,
    _other: [*mut T; 0],
}
pub unsafe fn shouldnt_work2<T: ?Sized>(from: *mut T) -> PtrAndEmptyArray<T> {
    std::mem::transmute(from)
    //~^ ERROR cannot transmute between types of different sizes, or dependently-sized types
}

fn main() {}
