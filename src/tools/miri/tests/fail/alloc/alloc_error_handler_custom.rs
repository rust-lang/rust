//@compile-flags: -Cpanic=abort
#![feature(start, core_intrinsics)]
#![feature(alloc_error_handler)]
#![feature(allocator_api)]
#![no_std]

extern crate alloc;

use alloc::alloc::*;
use core::fmt::Write;

#[path = "../../utils/mod.no_std.rs"]
mod utils;

#[alloc_error_handler]
fn alloc_error_handler(layout: Layout) -> ! {
    let _ = writeln!(utils::MiriStderr, "custom alloc error handler: {layout:?}");
    core::intrinsics::abort(); //~ERROR: aborted
}

// rustc requires us to provide some more things that aren't actually used by this test
mod plumbing {
    use super::*;

    #[panic_handler]
    fn panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    struct NoAlloc;

    unsafe impl GlobalAlloc for NoAlloc {
        unsafe fn alloc(&self, _: Layout) -> *mut u8 {
            unreachable!();
        }

        unsafe fn dealloc(&self, _: *mut u8, _: Layout) {
            unreachable!();
        }
    }

    #[global_allocator]
    static GLOBAL: NoAlloc = NoAlloc;
}

#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    handle_alloc_error(Layout::for_value(&0));
}
