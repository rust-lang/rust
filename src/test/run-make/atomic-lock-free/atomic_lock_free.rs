// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(cfg_target_has_atomic, no_core, intrinsics, lang_items)]
#![crate_type="rlib"]
#![no_core]

extern "rust-intrinsic" {
    fn atomic_xadd<T>(dst: *mut T, src: T) -> T;
}

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

#[cfg(target_has_atomic = "8")]
pub unsafe fn atomic_u8(x: *mut u8) {
    atomic_xadd(x, 1);
    atomic_xadd(x, 1);
}
#[cfg(target_has_atomic = "8")]
pub unsafe fn atomic_i8(x: *mut i8) {
    atomic_xadd(x, 1);
}
#[cfg(target_has_atomic = "16")]
pub unsafe fn atomic_u16(x: *mut u16) {
    atomic_xadd(x, 1);
}
#[cfg(target_has_atomic = "16")]
pub unsafe fn atomic_i16(x: *mut i16) {
    atomic_xadd(x, 1);
}
#[cfg(target_has_atomic = "32")]
pub unsafe fn atomic_u32(x: *mut u32) {
    atomic_xadd(x, 1);
}
#[cfg(target_has_atomic = "32")]
pub unsafe fn atomic_i32(x: *mut i32) {
    atomic_xadd(x, 1);
}
#[cfg(target_has_atomic = "64")]
pub unsafe fn atomic_u64(x: *mut u64) {
    atomic_xadd(x, 1);
}
#[cfg(target_has_atomic = "64")]
pub unsafe fn atomic_i64(x: *mut i64) {
    atomic_xadd(x, 1);
}
#[cfg(target_has_atomic = "ptr")]
pub unsafe fn atomic_usize(x: *mut usize) {
    atomic_xadd(x, 1);
}
#[cfg(target_has_atomic = "ptr")]
pub unsafe fn atomic_isize(x: *mut isize) {
    atomic_xadd(x, 1);
}
