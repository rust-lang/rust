#![feature(start, box_syntax, core_intrinsics, alloc_prelude, alloc_error_handler)]
#![no_std]

extern crate alloc;
extern crate alloc_system;

use alloc::prelude::v1::*;

use alloc_system::System;

#[global_allocator]
static ALLOC: System = System;

#[link(name = "c")]
extern "C" {
    fn puts(s: *const u8) -> i32;
}

#[panic_handler]
fn panic_handler(_: &core::panic::PanicInfo) -> ! {
    unsafe {
        core::intrinsics::abort();
    }
}

#[alloc_error_handler]
fn alloc_error_handler(_: alloc::alloc::Layout) -> ! {
    unsafe {
        core::intrinsics::abort();
    }
}

#[start]
fn main(_argc: isize, _argv: *const *const u8) -> isize {
    let world: Box<&str> = box "Hello World!\0";
    unsafe {
        puts(*world as *const str as *const u8);
    }

    0
}
