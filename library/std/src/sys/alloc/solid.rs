use super::{MIN_ALIGN, realloc_fallback};
use crate::alloc::Layout;

#[inline]
pub unsafe fn alloc(layout: Layout) -> *mut u8 {
    if layout.align() <= MIN_ALIGN && layout.align() <= layout.size() {
        unsafe { libc::malloc(layout.size()) as *mut u8 }
    } else {
        unsafe { libc::memalign(layout.align(), layout.size()) as *mut u8 }
    }
}

#[inline]
pub unsafe fn dealloc(ptr: *mut u8, _layout: Layout) {
    unsafe { libc::free(ptr as *mut libc::c_void) }
}

#[inline]
pub unsafe fn realloc(ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    unsafe {
        if layout.align() <= MIN_ALIGN && layout.align() <= new_size {
            libc::realloc(ptr as *mut libc::c_void, new_size) as *mut u8
        } else {
            realloc_fallback(ptr, layout, new_size)
        }
    }
}
