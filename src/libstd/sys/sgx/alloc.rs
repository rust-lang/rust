use crate::alloc::{GlobalAlloc, Layout, System};

use super::waitqueue::SpinMutex;

// Using a SpinMutex because we never want to exit the enclave waiting for the
// allocator.
#[cfg_attr(test, linkage = "available_externally")]
#[export_name = "_ZN16__rust_internals3std3sys3sgx5alloc8DLMALLOCE"]
static DLMALLOC: SpinMutex<dlmalloc::Dlmalloc> = SpinMutex::new(dlmalloc::DLMALLOC_INIT);

#[stable(feature = "alloc_system_type", since = "1.28.0")]
unsafe impl GlobalAlloc for System {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        DLMALLOC.lock().malloc(layout.size(), layout.align())
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        DLMALLOC.lock().calloc(layout.size(), layout.align())
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DLMALLOC.lock().free(ptr, layout.size(), layout.align())
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        DLMALLOC.lock().realloc(ptr, layout.size(), layout.align(), new_size)
    }
}
