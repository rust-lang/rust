#![feature(panic_handler, alloc_error_handler)]
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

#[panic_handler]
fn b(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
