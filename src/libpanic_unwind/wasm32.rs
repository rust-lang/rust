// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unwinding for wasm32
//!
//! Right now we don't support this, so this is just stubs

use alloc::boxed::Box;
use core::any::Any;
use core::intrinsics;

pub fn payload() -> *mut u8 {
    0 as *mut u8
}

pub unsafe fn cleanup(_ptr: *mut u8) -> Box<Any + Send> {
    intrinsics::abort()
}

pub unsafe fn panic(_data: Box<Any + Send>) -> u32 {
    intrinsics::abort()
}
