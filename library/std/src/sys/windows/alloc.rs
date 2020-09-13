#![deny(unsafe_op_in_unsafe_fn)]

use crate::alloc::{GlobalAlloc, Layout, System};
use crate::sys::c;
use crate::sys_common::alloc::{realloc_fallback, MIN_ALIGN};

#[repr(C)]
struct Header(*mut u8);

unsafe fn get_header<'a>(ptr: *mut u8) -> &'a mut Header {
    // SAFETY: the safety contract must be upheld by the caller
    unsafe { &mut *(ptr as *mut Header).offset(-1) }
}

unsafe fn align_ptr(ptr: *mut u8, align: usize) -> *mut u8 {
    // SAFETY: the safety contract must be upheld by the caller
    unsafe {
        let aligned = ptr.add(align - (ptr as usize & (align - 1)));
        *get_header(aligned) = Header(ptr);
        aligned
    }
}

#[inline]
unsafe fn allocate_with_flags(layout: Layout, flags: c::DWORD) -> *mut u8 {
    if layout.align() <= MIN_ALIGN {
        // SAFETY: `layout.size()` comes from `Layout` and is valid.
        return unsafe { c::HeapAlloc(c::GetProcessHeap(), flags, layout.size()) as *mut u8 };
    }

    let ptr = unsafe {
        // SAFETY: The caller must ensure that
        // `layout.size()` + `layout.size()` does not overflow.
        let size = layout.size() + layout.align();
        c::HeapAlloc(c::GetProcessHeap(), flags, size)
    };

    if ptr.is_null() {
        ptr as *mut u8
    } else {
        // SAFETY: `ptr` is a valid pointer
        // with enough allocated space to store the header.
        unsafe { align_ptr(ptr as *mut u8, layout.align()) }
    }
}

// SAFETY: All methods implemented follow the contract rules defined
// in `GlobalAlloc`.
#[stable(feature = "alloc_system_type", since = "1.28.0")]
unsafe impl GlobalAlloc for System {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the safety contract for `allocate_with_flags` must be upheld by the caller.
        unsafe { allocate_with_flags(layout, 0) }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the safety contract for `allocate_with_flags must be upheld by the caller.
        unsafe { allocate_with_flags(layout, c::HEAP_ZERO_MEMORY) }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: HeapFree is safe if ptr was allocated by this allocator
        let err = unsafe {
            if layout.align() <= MIN_ALIGN {
                c::HeapFree(c::GetProcessHeap(), 0, ptr as c::LPVOID);
            } else {
                let header = get_header(ptr);
                c::HeapFree(c::GetProcessHeap(), 0, header.0 as c::LPVOID);
            }
        }
        debug_assert!(err != 0, "Failed to free heap memory: {}", c::GetLastError());
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        unsafe {
            if layout.align() <= MIN_ALIGN {
                // SAFETY: HeapReAlloc is safe if ptr was allocated by this allocator
                // and new_size is not 0.
                unsafe {
                    c::HeapReAlloc(c::GetProcessHeap(), 0, ptr as c::LPVOID, new_size) as *mut u8
                }
            } else {
                // SAFETY: The safety contract for `realloc_fallback` must be upheld by the caller
                unsafe {
                    realloc_fallback(self, ptr, layout, new_size)
                }
            }
        }
    }
}
