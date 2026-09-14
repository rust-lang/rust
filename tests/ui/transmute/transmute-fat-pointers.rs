//@ normalize-stderr: "\d+ bits" -> "N bits"

// Tests that are conservative around thin/fat pointer mismatches.

#![allow(dead_code)]

use std::mem::transmute;
use std::ptr::NonNull;

fn a<T, U: ?Sized>(x: &[T]) -> &U {
    unsafe { transmute(x) } //~ ERROR cannot transmute between types of different sizes
}

fn b<T: ?Sized, U: ?Sized>(x: &T) -> &U {
    unsafe { transmute(x) } //~ ERROR cannot transmute between types of different sizes
}

fn c<T, U>(x: &T) -> &U {
    unsafe { transmute(x) } // Ok!
}

fn d<T, U>(x: &[T]) -> &[U] {
    unsafe { transmute(x) } // Ok!
}

fn e<T: ?Sized, U>(x: &T) -> &U {
    unsafe { transmute(x) } //~ ERROR cannot transmute between types of different sizes
}

fn f<T, U: ?Sized>(x: &T) -> &U {
    unsafe { transmute(x) } //~ ERROR cannot transmute between types of different sizes
}

// We can transmute even between pointers of unknown size as long as the metadata of
// input and output type are the same. This even accounts for null pointer optimizations.
fn g1<T: ?Sized>(x: &T) -> Option<&T> {
    unsafe { transmute(x) } // Ok!
}

fn g2<T: ?Sized>(x: Option<NonNull<T>>) -> NonNull<T> {
    unsafe { transmute(x) } // Ok!
}

fn g3<T: ?Sized>(x: *const T) -> Option<NonNull<T>> {
    unsafe { transmute(x) } // Ok!
}

// Make sure we can see through all the layers of `Box`.
fn h<T: ?Sized>(x: Box<T>) -> &'static T {
    unsafe { transmute(x) } // Ok!
}

// Make sure we can see through newtype wrappers.
struct Wrapper1<T>(T);

#[repr(C)]
struct Wrapper2<T>(T);

fn i1<T: ?Sized>(x: &T) -> Wrapper1<&T> {
    unsafe { transmute(x) } // Ok!
}
fn i2<T: ?Sized>(x: &T) -> Wrapper2<&T> {
    unsafe { transmute(x) } // Ok!
}

fn main() { }
