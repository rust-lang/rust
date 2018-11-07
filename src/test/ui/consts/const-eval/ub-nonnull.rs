// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs, const_transmute)]
#![allow(const_err)] // make sure we cannot allow away the errors tested here

use std::mem;
use std::ptr::NonNull;
use std::num::{NonZeroU8, NonZeroUsize};

const NULL_PTR: NonNull<u8> = unsafe { mem::transmute(0usize) };
//~^ ERROR it is undefined behavior to use this value

const NULL_U8: NonZeroU8 = unsafe { mem::transmute(0u8) };
//~^ ERROR it is undefined behavior to use this value
const NULL_USIZE: NonZeroUsize = unsafe { mem::transmute(0usize) };
//~^ ERROR it is undefined behavior to use this value

// Also test other uses of rustc_layout_scalar_valid_range_start

#[rustc_layout_scalar_valid_range_start(10)]
#[rustc_layout_scalar_valid_range_end(30)]
struct RestrictedRange1(u32);
const BAD_RANGE1: RestrictedRange1 = unsafe { RestrictedRange1(42) };
//~^ ERROR it is undefined behavior to use this value

#[rustc_layout_scalar_valid_range_start(30)]
#[rustc_layout_scalar_valid_range_end(10)]
struct RestrictedRange2(u32);
const BAD_RANGE2: RestrictedRange2 = unsafe { RestrictedRange2(20) };
//~^ ERROR it is undefined behavior to use this value

fn main() {}
