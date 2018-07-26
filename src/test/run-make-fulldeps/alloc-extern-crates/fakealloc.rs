// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type = "rlib"]
#![no_std]

#[inline]
pub unsafe fn allocate(_size: usize, _align: usize) -> *mut u8 { 0 as *mut u8 }

#[inline]
pub unsafe fn deallocate(_ptr: *mut u8, _old_size: usize, _align: usize) { }

#[inline]
pub unsafe fn reallocate(_ptr: *mut u8, _old_size: usize, _size: usize, _align: usize) -> *mut u8 {
    0 as *mut u8
}

#[inline]
pub unsafe fn reallocate_inplace(_ptr: *mut u8, old_size: usize, _size: usize,
                                    _align: usize) -> usize { old_size }

#[inline]
pub fn usable_size(size: usize, _align: usize) -> usize { size }

#[inline]
pub fn stats_print() { }
