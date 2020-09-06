#![deny(unsafe_op_in_unsafe_fn)]

use crate::alloc::{GlobalAlloc, Layout, System};
use crate::ptr;
use crate::sys_common::alloc::{realloc_fallback, MIN_ALIGN};

// SAFETY: All methods implemented follow the contract rules defined
// in `GlobalAlloc`.
#[stable(feature = "alloc_system_type", since = "1.28.0")]
unsafe impl GlobalAlloc for System {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.align() <= MIN_ALIGN && layout.align() <= layout.size() {
            // SAFETY: `libc::malloc` is guaranteed to be safe, it will allocate
            // `layout.size()` bytes of memory and return a pointer to it
            unsafe { libc::malloc(layout.size()) as *mut u8 }
        } else {
            // SAFETY: `libc::aligned_alloc` is guaranteed to be safe if
            // `layout.size()` is a multiple of `layout.align()`. This
            // constraint can be satisfied if `pad_to_align` is called,
            // which creates a layout by rounding the size of this layout up
            // to a multiple of the layout's alignment
            let aligned_layout = layout.pad_to_align();
            unsafe { libc::aligned_alloc(aligned_layout.align(), aligned_layout.size()) as *mut u8 }
        }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.align() <= MIN_ALIGN && layout.align() <= layout.size() {
            // SAFETY: `libc::calloc` is safe as long that `layout.size() * 1`
            // would not result in integer overflow which cannot happen,
            // multiplying by one never overflows
            unsafe { libc::calloc(layout.size(), 1) as *mut u8 }
        } else {
            // SAFETY: The safety contract for `alloc` must be upheld by the caller
            let ptr = unsafe { self.alloc(layout.clone()) };
            if !ptr.is_null() {
                // SAFETY: in the case of the `ptr` being not null
                // it will be properly aligned and a valid ptr
                // which satisfies `ptr::write_bytes` safety constrains
                unsafe { ptr::write_bytes(ptr, 0, layout.size()) };
            }
            ptr
        }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        // SAFETY: `libc::free` is guaranteed to be safe if `ptr` is allocated
        // by this allocator or if `ptr` is NULL
        unsafe { libc::free(ptr as *mut libc::c_void) }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if layout.align() <= MIN_ALIGN && layout.align() <= new_size {
            // SAFETY: `libc::realloc` is safe if `ptr` is allocated by this
            // allocator or NULL
            // - If `new_size` is 0 and `ptr` is not NULL, it will act as `libc::free`
            // - If `new_size` is not 0 and `ptr` is NULL, it will act as `libc::malloc`
            // - Else, it will resize the block accordingly
            unsafe { libc::realloc(ptr as *mut libc::c_void, new_size) as *mut u8 }
        } else {
            // SAFETY: The safety contract for `realloc_fallback` must be upheld by the caller
            unsafe { realloc_fallback(self, ptr, layout, new_size) }
        }
    }
}
