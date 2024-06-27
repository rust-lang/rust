//@ compile-flags: -Zinline-mir=yes --crate-type=lib

//@ known-bug: unknown
//@ build-fail
//@ failure-status: 101

use std::mem::ManuallyDrop;

pub struct Foo<T, const N: usize>([T; N]);

pub struct Dorp {}

impl Drop for Dorp {
    fn drop(&mut self) {}
}

#[inline]
// SAFETY: call this with a valid allocation idk
pub unsafe fn drop<const M: usize>(x: *mut Foo<Dorp, M>) {
    std::ptr::drop_in_place(x);
}
