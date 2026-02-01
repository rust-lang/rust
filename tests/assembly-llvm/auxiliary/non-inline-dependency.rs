//@ add-minicore
//@ no-prefer-dynamic
//@ compile-flags: --target nvptx64-nvidia-cuda -Ctarget-cpu=sm_30
//@ needs-llvm-components: nvptx
//@ ignore-backends: gcc
#![feature(no_core, intrinsics)]
#![crate_type = "rlib"]
#![no_core]

extern crate minicore;
use minicore::*;

#[rustc_intrinsic]
pub const fn wrapping_mul<T: Copy>(a: T, b: T) -> T;

#[rustc_intrinsic]
pub const fn mul_with_overflow<T: Copy>(x: T, y: T) -> (T, bool);

#[inline(never)]
#[no_mangle]
pub fn wrapping_external_fn(a: u32) -> u32 {
    wrapping_mul(a, a)
}

#[inline(never)]
#[no_mangle]
pub fn overflowing_external_fn(a: u32) -> u32 {
    mul_with_overflow(a, a).0
}
