// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(panic_implementation, alloc_error_handler)]
#![crate_type = "cdylib"]
#![no_std]

use core::alloc::*;

struct B;

unsafe impl GlobalAlloc for B {
    unsafe fn alloc(&self, x: Layout) -> *mut u8 {
        1 as *mut u8
    }

    unsafe fn dealloc(&self, ptr: *mut u8, x: Layout) {
    }
}

#[global_allocator]
static A: B = B;

#[no_mangle]
pub extern fn foo(a: u32) -> u32 {
    assert_eq!(a, 3);
    a * 2
}

#[alloc_error_handler]
fn a(_: core::alloc::Layout) -> ! {
    loop {}
}

#[panic_implementation]
fn b(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
