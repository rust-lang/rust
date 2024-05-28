use crate::{
    alloc::{GlobalAlloc, Layout, System},
    ptr,
    sync::atomic::{AtomicBool, Ordering},
};

static mut DLMALLOC: dlmalloc::Dlmalloc<Vexos> = dlmalloc::Dlmalloc::new_with_allocator(Vexos);

extern "C" {
    static mut __heap_start: u8;
    static mut __heap_end: u8;
}

struct Vexos;

unsafe impl dlmalloc::Allocator for Vexos {
    /// Allocs system resources
    fn alloc(&self, _size: usize) -> (*mut u8, usize, u32) {
        static INIT: AtomicBool = AtomicBool::new(false);

        if !INIT.swap(true, Ordering::Relaxed) {
            unsafe {
                (
                    ptr::addr_of_mut!(__heap_start).cast(),
                    ptr::addr_of!(__heap_end).byte_offset_from(ptr::addr_of!(__heap_start)) as _,
                    0,
                )
            }
        } else {
            (ptr::null_mut(), 0, 0)
        }
    }

    fn remap(&self, _ptr: *mut u8, _oldsize: usize, _newsize: usize, _can_move: bool) -> *mut u8 {
        ptr::null_mut()
    }

    fn free_part(&self, _ptr: *mut u8, _oldsize: usize, _newsize: usize) -> bool {
        false
    }

    fn free(&self, _ptr: *mut u8, _size: usize) -> bool {
        return false;
    }

    fn can_release_part(&self, _flags: u32) -> bool {
        false
    }

    fn allocates_zeros(&self) -> bool {
        false
    }

    fn page_size(&self) -> usize {
        0x1000
    }
}

#[stable(feature = "alloc_system_type", since = "1.28.0")]
unsafe impl GlobalAlloc for System {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: DLMALLOC access is guaranteed to be safe because the lock gives us unique and non-reentrant access.
        // Calling malloc() is safe because preconditions on this function match the trait method preconditions.
        let _lock = lock::lock();
        unsafe { DLMALLOC.malloc(layout.size(), layout.align()) }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: DLMALLOC access is guaranteed to be safe because the lock gives us unique and non-reentrant access.
        // Calling calloc() is safe because preconditions on this function match the trait method preconditions.
        let _lock = lock::lock();
        unsafe { DLMALLOC.calloc(layout.size(), layout.align()) }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: DLMALLOC access is guaranteed to be safe because the lock gives us unique and non-reentrant access.
        // Calling free() is safe because preconditions on this function match the trait method preconditions.
        let _lock = lock::lock();
        unsafe { DLMALLOC.free(ptr, layout.size(), layout.align()) }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: DLMALLOC access is guaranteed to be safe because the lock gives us unique and non-reentrant access.
        // Calling realloc() is safe because preconditions on this function match the trait method preconditions.
        let _lock = lock::lock();
        unsafe { DLMALLOC.realloc(ptr, layout.size(), layout.align(), new_size) }
    }
}

mod lock {
    #[inline]
    pub fn lock() {} // we don't have threads, which makes this real easy
}
