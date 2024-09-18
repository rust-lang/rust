//! Helper for 'no-allocation-before-main'.
//!
//! This also contains a meta-test to make sure that the AbortingAllocator does indeed abort.
//!
//! -Cprefer-dynamic=no is required as otherwise #[global_allocator] does nothing.
//@ run-fail
//@ compile-flags: -Cprefer-dynamic=no

use std::{sync::atomic::{AtomicBool, Ordering}, alloc::System};

static ABORT: AtomicBool = AtomicBool::new(true);

pub struct AbortingAllocator(System);

unsafe impl std::alloc::GlobalAlloc for AbortingAllocator {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        if ABORT.swap(false, Ordering::SeqCst) {
            println!("{}", std::backtrace::Backtrace::force_capture());
            std::process::abort();
        }

        self.0.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        if ABORT.swap(false, Ordering::SeqCst) {
            println!("{}", std::backtrace::Backtrace::force_capture());
            std::process::abort();
        }

        self.0.dealloc(ptr, layout)
    }
}

#[global_allocator]
static ALLOCATOR: AbortingAllocator = AbortingAllocator(System);

fn main() {
    std::hint::black_box(String::from("An allocation"));
}
