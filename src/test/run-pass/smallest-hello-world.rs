// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Smallest "hello world" with a libc runtime

// ignore-windows
// ignore-android

#![feature(intrinsics, lang_items, start, no_core, alloc_system)]
#![feature(global_allocator, allocator_api)]
#![no_std]

extern crate alloc_system;

use alloc_system::System;

#[global_allocator]
static A: System = System;

extern {
    fn puts(s: *const u8);
}

#[no_mangle]
#[lang = "eh_personality"] pub extern fn rust_eh_personality() {}
#[lang = "panic_fmt"] fn panic_fmt() -> ! { loop {} }

#[start]
fn main(_: isize, _: *const *const u8) -> isize {
    unsafe {
        puts("Hello!\0".as_ptr() as *const u8);
    }
    return 0
}
