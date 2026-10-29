use crate::alloc::Layout;
use crate::sys::pal::abi;

#[inline]
pub unsafe fn alloc(layout: Layout) -> *mut u8 {
    unsafe { abi::sys_alloc_aligned(layout.size(), layout.align()) }
}

#[inline]
pub unsafe fn dealloc(_ptr: *mut u8, _layout: Layout) {
    // this allocator never deallocates memory
}

#[inline]
pub unsafe fn realloc(ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    // SAFETY: this is just a `pub` wrapper.
    unsafe { super::realloc_fallback(ptr, layout, new_size) }
}
