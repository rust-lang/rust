use crate::alloc::{GlobalAlloc, Layout, System};
use crate::sys::c;
use crate::sys_common::alloc::{MIN_ALIGN, realloc_fallback};

#[repr(C)]
struct Header(*mut u8);

unsafe fn get_header<'a>(ptr: *mut u8) -> &'a mut Header {
    &mut *(ptr as *mut Header).offset(-1)
}

unsafe fn align_ptr(ptr: *mut u8, align: usize) -> *mut u8 {
    let aligned = ptr.add(align - (ptr as usize & (align - 1)));
    *get_header(aligned) = Header(ptr);
    aligned
}

#[inline]
unsafe fn allocate_with_flags(layout: Layout, flags: c::DWORD) -> *mut u8 {
    if layout.align() <= MIN_ALIGN {
        return c::HeapAlloc(c::GetProcessHeap(), flags, layout.size()) as *mut u8
    }

    let size = layout.size() + layout.align();
    let ptr = c::HeapAlloc(c::GetProcessHeap(), flags, size);
    if ptr.is_null() {
        ptr as *mut u8
    } else {
        align_ptr(ptr as *mut u8, layout.align())
    }
}

#[stable(feature = "alloc_system_type", since = "1.28.0")]
unsafe impl GlobalAlloc for System {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        allocate_with_flags(layout, 0)
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        allocate_with_flags(layout, c::HEAP_ZERO_MEMORY)
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if layout.align() <= MIN_ALIGN {
            let err = c::HeapFree(c::GetProcessHeap(), 0, ptr as c::LPVOID);
            debug_assert!(err != 0, "Failed to free heap memory: {}",
                          c::GetLastError());
        } else {
            let header = get_header(ptr);
            let err = c::HeapFree(c::GetProcessHeap(), 0, header.0 as c::LPVOID);
            debug_assert!(err != 0, "Failed to free heap memory: {}",
                          c::GetLastError());
        }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if layout.align() <= MIN_ALIGN {
            c::HeapReAlloc(c::GetProcessHeap(), 0, ptr as c::LPVOID, new_size) as *mut u8
        } else {
            realloc_fallback(self, ptr, layout, new_size)
        }
    }
}
