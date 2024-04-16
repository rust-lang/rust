//@compile-flags: -Cpanic=abort
#![feature(start, core_intrinsics)]
#![feature(alloc_error_handler)]
#![feature(allocator_api)]
#![no_std]

extern crate alloc;

use alloc::alloc::*;
use alloc::boxed::Box;
use core::ptr::NonNull;

struct BadAlloc;

// Create a failing allocator; that is the only way to actually call the alloc error handler.
unsafe impl Allocator for BadAlloc {
    fn allocate(&self, _l: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Err(AllocError)
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {
        unreachable!();
    }
}

#[alloc_error_handler]
fn alloc_error_handler(_: Layout) -> ! {
    extern "Rust" {
        fn miri_write_to_stderr(bytes: &[u8]);
    }
    let msg = "custom alloc error handler called!\n";
    unsafe { miri_write_to_stderr(msg.as_bytes()) };
    core::intrinsics::abort(); //~ERROR: aborted
}

// rustc requires us to provide some more things that aren't actually used by this test
mod plumbing {
    use super::*;

    #[panic_handler]
    fn panic_handler(_: &core::panic::PanicInfo) -> ! {
        core::intrinsics::abort();
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
    let _b = Box::new_in(0, BadAlloc);
    0
}
