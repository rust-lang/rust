//! Global Allocator for ThingOS.
//!
//! This is a stub — Thing-OS userspace apps use stem's `#[global_allocator]`
//! which takes precedence over this `System` allocator. These methods should
//! never actually be called, but they must exist for std to compile.

use crate::alloc::{GlobalAlloc, Layout, System};

#[stable(feature = "alloc_system_type", since = "1.28.0")]
unsafe impl GlobalAlloc for System {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // stem's global allocator handles all allocations.
        let _ = layout;
        crate::ptr::null_mut()
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        // stem's global allocator handles all deallocations.
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        self.alloc(layout)
    }

    unsafe fn realloc(&self, _ptr: *mut u8, _old_layout: Layout, _new_size: usize) -> *mut u8 {
        core::ptr::null_mut()
    }
}
