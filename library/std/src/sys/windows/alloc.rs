#![deny(unsafe_op_in_unsafe_fn)]

use crate::alloc::{GlobalAlloc, Layout, System};
use crate::ffi::c_void;
use crate::ptr;
use crate::sync::atomic::{AtomicPtr, Ordering};
use crate::sys::c;
use crate::sys::common::alloc::{realloc_fallback, MIN_ALIGN};

#[cfg(test)]
mod tests;

// Heap memory management on Windows is done by using the system Heap API (heapapi.h)
// See https://docs.microsoft.com/windows/win32/api/heapapi/

// Flag to indicate that the memory returned by `HeapAlloc` should be zeroed.
const HEAP_ZERO_MEMORY: c::DWORD = 0x00000008;

#[link(name = "kernel32")]
extern "system" {
    // Get a handle to the default heap of the current process, or null if the operation fails.
    //
    // SAFETY: Successful calls to this function within the same process are assumed to
    // always return the same handle, which remains valid for the entire lifetime of the process.
    //
    // See https://docs.microsoft.com/windows/win32/api/heapapi/nf-heapapi-getprocessheap
    fn GetProcessHeap() -> c::HANDLE;

    // Allocate a block of `dwBytes` bytes of memory from a given heap `hHeap`.
    // The allocated memory may be uninitialized, or zeroed if `dwFlags` is
    // set to `HEAP_ZERO_MEMORY`.
    //
    // Returns a pointer to the newly-allocated memory or null if the operation fails.
    // The returned pointer will be aligned to at least `MIN_ALIGN`.
    //
    // SAFETY:
    //  - `hHeap` must be a non-null handle returned by `GetProcessHeap`.
    //  - `dwFlags` must be set to either zero or `HEAP_ZERO_MEMORY`.
    //
    // Note that `dwBytes` is allowed to be zero, contrary to some other allocators.
    //
    // See https://docs.microsoft.com/windows/win32/api/heapapi/nf-heapapi-heapalloc
    fn HeapAlloc(hHeap: c::HANDLE, dwFlags: c::DWORD, dwBytes: c::SIZE_T) -> c::LPVOID;

    // Reallocate a block of memory behind a given pointer `lpMem` from a given heap `hHeap`,
    // to a block of at least `dwBytes` bytes, either shrinking the block in place,
    // or allocating at a new location, copying memory, and freeing the original location.
    //
    // Returns a pointer to the reallocated memory or null if the operation fails.
    // The returned pointer will be aligned to at least `MIN_ALIGN`.
    // If the operation fails the given block will never have been freed.
    //
    // SAFETY:
    //  - `hHeap` must be a non-null handle returned by `GetProcessHeap`.
    //  - `dwFlags` must be set to zero.
    //  - `lpMem` must be a non-null pointer to an allocated block returned by `HeapAlloc` or
    //     `HeapReAlloc`, that has not already been freed.
    // If the block was successfully reallocated at a new location, pointers pointing to
    // the freed memory, such as `lpMem`, must not be dereferenced ever again.
    //
    // Note that `dwBytes` is allowed to be zero, contrary to some other allocators.
    //
    // See https://docs.microsoft.com/windows/win32/api/heapapi/nf-heapapi-heaprealloc
    fn HeapReAlloc(
        hHeap: c::HANDLE,
        dwFlags: c::DWORD,
        lpMem: c::LPVOID,
        dwBytes: c::SIZE_T,
    ) -> c::LPVOID;

    // Free a block of memory behind a given pointer `lpMem` from a given heap `hHeap`.
    // Returns a nonzero value if the operation is successful, and zero if the operation fails.
    //
    // SAFETY:
    //  - `hHeap` must be a non-null handle returned by `GetProcessHeap`.
    //  - `dwFlags` must be set to zero.
    //  - `lpMem` must be a pointer to an allocated block returned by `HeapAlloc` or `HeapReAlloc`,
    //     that has not already been freed.
    // If the block was successfully freed, pointers pointing to the freed memory, such as `lpMem`,
    // must not be dereferenced ever again.
    //
    // Note that `lpMem` is allowed to be null, which will not cause the operation to fail.
    //
    // See https://docs.microsoft.com/windows/win32/api/heapapi/nf-heapapi-heapfree
    fn HeapFree(hHeap: c::HANDLE, dwFlags: c::DWORD, lpMem: c::LPVOID) -> c::BOOL;
}

// Cached handle to the default heap of the current process.
// Either a non-null handle returned by `GetProcessHeap`, or null when not yet initialized or `GetProcessHeap` failed.
static HEAP: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());

// Get a handle to the default heap of the current process, or null if the operation fails.
// If this operation is successful, `HEAP` will be successfully initialized and contain
// a non-null handle returned by `GetProcessHeap`.
#[inline]
fn init_or_get_process_heap() -> c::HANDLE {
    let heap = HEAP.load(Ordering::Relaxed);
    if heap.is_null() {
        // `HEAP` has not yet been successfully initialized
        let heap = unsafe { GetProcessHeap() };
        if !heap.is_null() {
            // SAFETY: No locking is needed because within the same process,
            // successful calls to `GetProcessHeap` will always return the same value, even on different threads.
            HEAP.store(heap, Ordering::Release);

            // SAFETY: `HEAP` contains a non-null handle returned by `GetProcessHeap`
            heap
        } else {
            // Could not get the current process heap.
            ptr::null_mut()
        }
    } else {
        // SAFETY: `HEAP` contains a non-null handle returned by `GetProcessHeap`
        heap
    }
}

// Get a non-null handle to the default heap of the current process.
// SAFETY: `HEAP` must have been successfully initialized.
#[inline]
unsafe fn get_process_heap() -> c::HANDLE {
    HEAP.load(Ordering::Acquire)
}

// Header containing a pointer to the start of an allocated block.
// SAFETY: Size and alignment must be <= `MIN_ALIGN`.
#[repr(C)]
struct Header(*mut u8);

// Allocate a block of optionally zeroed memory for a given `layout`.
// SAFETY: Returns a pointer satisfying the guarantees of `System` about allocated pointers,
// or null if the operation fails. If this returns non-null `HEAP` will have been successfully
// initialized.
#[inline]
unsafe fn allocate(layout: Layout, zeroed: bool) -> *mut u8 {
    let heap = init_or_get_process_heap();
    if heap.is_null() {
        // Allocation has failed, could not get the current process heap.
        return ptr::null_mut();
    }

    // Allocated memory will be either zeroed or uninitialized.
    let flags = if zeroed { HEAP_ZERO_MEMORY } else { 0 };

    if layout.align() <= MIN_ALIGN {
        // SAFETY: `heap` is a non-null handle returned by `GetProcessHeap`.
        // The returned pointer points to the start of an allocated block.
        unsafe { HeapAlloc(heap, flags, layout.size()) as *mut u8 }
    } else {
        // Allocate extra padding in order to be able to satisfy the alignment.
        let total = layout.align() + layout.size();

        // SAFETY: `heap` is a non-null handle returned by `GetProcessHeap`.
        let ptr = unsafe { HeapAlloc(heap, flags, total) as *mut u8 };
        if ptr.is_null() {
            // Allocation has failed.
            return ptr::null_mut();
        }

        // Create a correctly aligned pointer offset from the start of the allocated block,
        // and write a header before it.

        let offset = layout.align() - (ptr.addr() & (layout.align() - 1));
        // SAFETY: `MIN_ALIGN` <= `offset` <= `layout.align()` and the size of the allocated
        // block is `layout.align() + layout.size()`. `aligned` will thus be a correctly aligned
        // pointer inside the allocated block with at least `layout.size()` bytes after it and at
        // least `MIN_ALIGN` bytes of padding before it.
        let aligned = unsafe { ptr.add(offset) };
        // SAFETY: Because the size and alignment of a header is <= `MIN_ALIGN` and `aligned`
        // is aligned to at least `MIN_ALIGN` and has at least `MIN_ALIGN` bytes of padding before
        // it, it is safe to write a header directly before it.
        unsafe { ptr::write((aligned as *mut Header).sub(1), Header(ptr)) };

        // SAFETY: The returned pointer does not point to the to the start of an allocated block,
        // but there is a header readable directly before it containing the location of the start
        // of the block.
        aligned
    }
}

// All pointers returned by this allocator have, in addition to the guarantees of `GlobalAlloc`, the
// following properties:
//
// If the pointer was allocated or reallocated with a `layout` specifying an alignment <= `MIN_ALIGN`
// the pointer will be aligned to at least `MIN_ALIGN` and point to the start of the allocated block.
//
// If the pointer was allocated or reallocated with a `layout` specifying an alignment > `MIN_ALIGN`
// the pointer will be aligned to the specified alignment and not point to the start of the allocated block.
// Instead there will be a header readable directly before the returned pointer, containing the actual
// location of the start of the block.
#[stable(feature = "alloc_system_type", since = "1.28.0")]
unsafe impl GlobalAlloc for System {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: Pointers returned by `allocate` satisfy the guarantees of `System`
        let zeroed = false;
        unsafe { allocate(layout, zeroed) }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: Pointers returned by `allocate` satisfy the guarantees of `System`
        let zeroed = true;
        unsafe { allocate(layout, zeroed) }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let block = {
            if layout.align() <= MIN_ALIGN {
                ptr
            } else {
                // The location of the start of the block is stored in the padding before `ptr`.

                // SAFETY: Because of the contract of `System`, `ptr` is guaranteed to be non-null
                // and have a header readable directly before it.
                unsafe { ptr::read((ptr as *mut Header).sub(1)).0 }
            }
        };

        // SAFETY: because `ptr` has been successfully allocated with this allocator,
        // `HEAP` must have been successfully initialized.
        let heap = unsafe { get_process_heap() };

        // SAFETY: `heap` is a non-null handle returned by `GetProcessHeap`,
        // `block` is a pointer to the start of an allocated block.
        unsafe { HeapFree(heap, 0, block as c::LPVOID) };
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if layout.align() <= MIN_ALIGN {
            // SAFETY: because `ptr` has been successfully allocated with this allocator,
            // `HEAP` must have been successfully initialized.
            let heap = unsafe { get_process_heap() };

            // SAFETY: `heap` is a non-null handle returned by `GetProcessHeap`,
            // `ptr` is a pointer to the start of an allocated block.
            // The returned pointer points to the start of an allocated block.
            unsafe { HeapReAlloc(heap, 0, ptr as c::LPVOID, new_size) as *mut u8 }
        } else {
            // SAFETY: `realloc_fallback` is implemented using `dealloc` and `alloc`, which will
            // correctly handle `ptr` and return a pointer satisfying the guarantees of `System`
            unsafe { realloc_fallback(self, ptr, layout, new_size) }
        }
    }
}
