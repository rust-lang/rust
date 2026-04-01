//@ no-prefer-dynamic

#![feature(allocator_api)]
#![crate_type = "rlib"]

use std::alloc::{GlobalAlloc, System, Layout};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct A(pub AtomicUsize);

unsafe impl GlobalAlloc for A {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.0.fetch_add(1, Ordering::SeqCst);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.0.fetch_add(1, Ordering::SeqCst);
        System.dealloc(ptr, layout)
    }
}
