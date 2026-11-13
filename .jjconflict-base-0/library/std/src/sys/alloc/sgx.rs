use crate::alloc::Layout;
use crate::ptr;
use crate::sync::atomic::{Atomic, AtomicBool, Ordering};
use crate::sys::pal::abi::mem as sgx_mem;
use crate::sys::pal::waitqueue::SpinMutex;

// Using a SpinMutex because we never want to exit the enclave waiting for the
// allocator.
//
// The current allocator here is the `dlmalloc` crate which we've got included
// in the rust-lang/rust repository as a submodule. The crate is a port of
// dlmalloc.c from C to Rust.
//
// Specifying linkage/symbol name is solely to ensure a single instance between this crate and its unit tests
#[cfg_attr(test, linkage = "available_externally")]
#[unsafe(export_name = "_ZN16__rust_internals3std3sys5alloc3sgx8DLMALLOCE")]
static DLMALLOC: SpinMutex<dlmalloc::Dlmalloc<Sgx>> =
    SpinMutex::new(dlmalloc::Dlmalloc::new_with_allocator(Sgx {}));

struct Sgx;

unsafe impl dlmalloc::Allocator for Sgx {
    /// Allocs system resources
    fn alloc(&self, _size: usize) -> (*mut u8, usize, u32) {
        static INIT: Atomic<bool> = AtomicBool::new(false);

        // No ordering requirement since this function is protected by the global lock.
        if !INIT.swap(true, Ordering::Relaxed) {
            (sgx_mem::heap_base() as _, sgx_mem::heap_size(), 0)
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

#[inline]
pub unsafe fn alloc(layout: Layout) -> *mut u8 {
    // SAFETY: the caller must uphold the safety contract for `malloc`
    unsafe { DLMALLOC.lock().malloc(layout.size(), layout.align()) }
}

#[inline]
pub unsafe fn alloc_zeroed(layout: Layout) -> *mut u8 {
    // SAFETY: the caller must uphold the safety contract for `malloc`
    unsafe { DLMALLOC.lock().calloc(layout.size(), layout.align()) }
}

#[inline]
pub unsafe fn dealloc(ptr: *mut u8, layout: Layout) {
    // SAFETY: the caller must uphold the safety contract for `malloc`
    unsafe { DLMALLOC.lock().free(ptr, layout.size(), layout.align()) }
}

#[inline]
pub unsafe fn realloc(ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    // SAFETY: the caller must uphold the safety contract for `malloc`
    unsafe { DLMALLOC.lock().realloc(ptr, layout.size(), layout.align(), new_size) }
}

// The following functions are needed by libunwind. These symbols are named
// in pre-link args for the target specification, so keep that in sync.
#[cfg(not(test))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __rust_c_alloc(size: usize, align: usize) -> *mut u8 {
    unsafe { crate::alloc::alloc(Layout::from_size_align_unchecked(size, align)) }
}

#[cfg(not(test))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __rust_c_dealloc(ptr: *mut u8, size: usize, align: usize) {
    unsafe { crate::alloc::dealloc(ptr, Layout::from_size_align_unchecked(size, align)) }
}
