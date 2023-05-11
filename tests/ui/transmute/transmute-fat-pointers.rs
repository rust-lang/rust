// normalize-stderr-test "\d+ bits" -> "N bits"

// Tests that are conservative around thin/fat pointer mismatches.

#![allow(dead_code)]

use std::mem::transmute;

fn a<T, U: ?Sized>(x: &[T]) -> &U {
    unsafe { transmute(x) } //~ ERROR cannot transmute between types of different sizes
}

fn b<T: ?Sized, U: ?Sized>(x: &T) -> &U {
    unsafe { transmute(x) } //~ ERROR cannot transmute between types of different sizes
}

fn c<T, U>(x: &T) -> &U {
    unsafe { transmute(x) }
}

fn d<T, U>(x: &[T]) -> &[U] {
    unsafe { transmute(x) }
}

fn e<T: ?Sized, U>(x: &T) -> &U {
    unsafe { transmute(x) } //~ ERROR cannot transmute between types of different sizes
}

fn f<T, U: ?Sized>(x: &T) -> &U {
    unsafe { transmute(x) } //~ ERROR cannot transmute between types of different sizes
}

fn main() { }
