//@ build-fail
//@ compile-flags: --target i686-unknown-linux-gnu --crate-type lib
//@ needs-llvm-components: x86
//@ error-pattern: too big for the current architecture
#![feature(no_core, lang_items, intrinsics)]
#![allow(internal_features)]
#![no_std]
#![no_core]

#[lang = "sized"]
pub trait Sized {}
#[lang = "copy"]
pub trait Copy: Sized {}

// 0x7fffffff is fine, but with the padding for the unsized tail it is too big.
#[repr(C)]
pub struct Example([u8; 0x7fffffff], [u16]);

extern "rust-intrinsic" {
    pub fn size_of_val<T: ?Sized>(_: *const T) -> usize;
}

// We guarantee that with length 0, `size_of_val_raw` (which calls the `size_of_val` intrinsic)
// is safe to call. The compiler aborts execution if a length of 0 would overflow.
// So let's construct a case where length 0 just barely overflows, and ensure that
// does abort execution.
pub fn check(x: *const Example) {
    unsafe { size_of_val(x); }
}
