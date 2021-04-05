// build-pass
// compile-flags: -C panic=unwind -C target-feature=+crt-static
// only-msvc
// Test that `no_std` with `panic=unwind` under MSVC toolchain
// doesn't cause error when linking to libcmt.

#![no_std]
#![no_main]
#![feature(alloc_error_handler)]
#![feature(panic_unwind)]

use core::alloc::{GlobalAlloc, Layout};

struct DummyAllocator;

unsafe impl GlobalAlloc for DummyAllocator {
    unsafe fn alloc(&self, _layout: Layout) -> *mut u8 {
        core::ptr::null_mut()
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
}

#[global_allocator]
static ALLOC: DummyAllocator = DummyAllocator;

#[alloc_error_handler]
fn rust_oom(_layout: Layout) -> ! {
    panic!()
}

extern crate panic_unwind;

use core::panic::PanicInfo;

#[panic_handler]
fn handle_panic(_: &PanicInfo) -> ! {
    loop {}
}

#[link(name = "libcmt")]
extern "C" {}

#[no_mangle]
pub extern "C" fn main() -> i32 {
    panic!();
}
