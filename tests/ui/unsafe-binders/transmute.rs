//@ compile-flags: -Znext-solver
//@ check-fail
//@ known-bug: #130516

// Transmuting between `&T` and `unsafe<'a> &'a T` works for any `T`.

#![feature(unsafe_binders)]
#![allow(incomplete_features)]

use std::mem::transmute;

fn into_unsized<T: ?Sized>(x: &T) -> unsafe<'a> &'a T {
    unsafe { transmute(x) }
}

fn out_of_unsized<T: ?Sized>(x: unsafe<'a> &'a T) -> &'static T {
    unsafe { transmute(x) }
}

fn into_sized<T>(x: &T) -> unsafe<'a> &'a T {
    unsafe { transmute(x) }
}

fn into_slice<T>(x: &[T]) -> unsafe<'a> &'a [T] {
    unsafe { transmute(x) }
}

fn into_concrete(x: &u8) -> unsafe<'a> &'a u8 {
    unsafe { transmute(x) }
}

fn main() {}
