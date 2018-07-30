// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C panic=abort
// no-prefer-dynamic

#![no_std]
#![crate_type = "staticlib"]
#![feature(panic_implementation, alloc_error_handler, alloc)]

#[panic_implementation]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

extern crate alloc;

#[global_allocator]
static A: MyAlloc = MyAlloc;

struct MyAlloc;

unsafe impl core::alloc::GlobalAlloc for MyAlloc {
    unsafe fn alloc(&self, _: core::alloc::Layout) -> *mut u8 { 0 as _ }
    unsafe fn dealloc(&self, _: *mut u8, _: core::alloc::Layout) {}
}
