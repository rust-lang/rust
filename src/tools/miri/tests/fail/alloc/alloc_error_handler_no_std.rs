//@compile-flags: -Cpanic=abort
#![feature(core_intrinsics)]
#![feature(alloc_error_handler)]
#![feature(allocator_api)]
#![no_std]
#![no_main]

extern crate alloc;

use alloc::alloc::*;
use core::fmt::Write;

#[path = "../../utils/mod.no_std.rs"]
mod utils;

// The default no_std alloc_error_handler is a panic.

#[panic_handler]
fn panic_handler(panic_info: &core::panic::PanicInfo) -> ! {
    let _ = writeln!(utils::MiriStderr, "custom panic handler called!");
    let _ = writeln!(utils::MiriStderr, "{panic_info}");
    core::intrinsics::abort(); //~ERROR: aborted
}

// rustc requires us to provide some more things that aren't actually used by this test
mod plumbing {
    use super::*;

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

#[no_mangle]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    handle_alloc_error(Layout::for_value(&0));
}
