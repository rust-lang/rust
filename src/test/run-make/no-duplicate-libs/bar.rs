// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![no_std]
#![feature(lang_items)]
#![crate_type = "dylib"]

extern crate libc;

#[no_mangle]
pub extern fn bar() {}

#[lang = "stack_exhausted"] fn stack_exhausted() {}
#[lang = "eh_personality"] fn eh_personality() {}
#[lang = "fail_fmt"] fn fail_fmt() -> ! { loop {} }
