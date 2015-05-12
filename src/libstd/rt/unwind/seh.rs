// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::v1::*;

use any::Any;
use intrinsics;
use libc::c_void;

pub unsafe fn panic(_data: Box<Any + Send + 'static>) -> ! {
    intrinsics::abort();
}

pub unsafe fn cleanup(_ptr: *mut c_void) -> Box<Any + Send + 'static> {
    intrinsics::abort();
}

#[lang = "eh_personality"]
#[no_mangle]
pub extern fn rust_eh_personality() {}

#[no_mangle]
pub extern fn rust_eh_personality_catch() {}
