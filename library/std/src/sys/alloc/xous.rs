// FIXME(static_mut_refs): Do not allow `static_mut_refs` lint
#![allow(static_mut_refs)]

use crate::alloc::{GlobalAlloc, Layout, System};

#[cfg(not(test))]
#[export_name = "_ZN16__rust_internals3std3sys4xous5alloc8DLMALLOCE"]
static mut DLMALLOC: dlmalloc::Dlmalloc = dlmalloc::Dlmalloc::new();

#[cfg(test)]
extern "Rust" {
    #[link_name = "_ZN16__rust_internals3std3sys4xous5alloc8DLMALLOCE"]
    static mut DLMALLOC: dlmalloc::Dlmalloc;
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
    use crate::sync::atomic::AtomicI32;
    use crate::sync::atomic::Ordering::{Acquire, Release};

    static LOCKED: AtomicI32 = AtomicI32::new(0);

    pub struct DropLock;

    pub fn lock() -> DropLock {
        loop {
            if LOCKED.swap(1, Acquire) == 0 {
                return DropLock;
            }
            crate::os::xous::ffi::do_yield();
        }
    }

    impl Drop for DropLock {
        fn drop(&mut self) {
            let r = LOCKED.swap(0, Release);
            debug_assert_eq!(r, 1);
        }
    }
}
