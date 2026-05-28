use crate::alloc::{GlobalAlloc, Layout, System};

#[stable(feature = "alloc_system_type", since = "1.28.0")]
unsafe impl GlobalAlloc for System {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: same requirements as in GlobalAlloc::alloc.
        moto_rt::alloc::alloc(layout)
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: same requirements as in GlobalAlloc::alloc_zeroed.
        moto_rt::alloc::alloc_zeroed(layout)
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: same requirements as in GlobalAlloc::dealloc.
        unsafe { moto_rt::alloc::dealloc(ptr, layout) }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: same requirements as in GlobalAlloc::realloc.
        unsafe { moto_rt::alloc::realloc(ptr, layout, new_size) }
    }
}
