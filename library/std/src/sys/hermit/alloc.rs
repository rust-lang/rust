#![deny(unsafe_op_in_unsafe_fn)]

use crate::alloc::{GlobalAlloc, Layout, System};
use crate::ptr;
use crate::sys::hermit::abi;

#[stable(feature = "alloc_system_type", since = "1.28.0")]
unsafe impl GlobalAlloc for System {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The safety contract for `malloc` must be upheld by the caller.
        unsafe { abi::malloc(layout.size(), layout.align()) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The safety contract for `malloc` must be upheld by the caller.
        // Also, `addr` must be valid for writes of `layout.size() * size_of::<u8>()` bytes.
        unsafe {
            let addr = abi::malloc(layout.size(), layout.align());

            if !addr.is_null() {
                ptr::write_bytes(addr, 0x00, layout.size());
            }

            addr
        }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: The safety contract for `free` must be upheld by the caller.
        unsafe { abi::free(ptr, layout.size(), layout.align()) }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: The safety contract for `realloc` must be upheld by the caller.
        unsafe { abi::realloc(ptr, layout.size(), layout.align(), new_size) }
    }
}
