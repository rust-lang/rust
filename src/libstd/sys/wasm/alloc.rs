//! This is an implementation of a global allocator on the wasm32 platform when
//! emscripten is not in use. In that situation there's no actual runtime for us
//! to lean on for allocation, so instead we provide our own!
//!
//! The wasm32 instruction set has two instructions for getting the current
//! amount of memory and growing the amount of memory. These instructions are the
//! foundation on which we're able to build an allocator, so we do so! Note that
//! the instructions are also pretty "global" and this is the "global" allocator
//! after all!
//!
//! The current allocator here is the `dlmalloc` crate which we've got included
//! in the rust-lang/rust repository as a submodule. The crate is a port of
//! dlmalloc.c from C to Rust and is basically just so we can have "pure Rust"
//! for now which is currently technically required (can't link with C yet).
//!
//! The crate itself provides a global allocator which on wasm has no
//! synchronization as there are no threads!

use crate::alloc::{GlobalAlloc, Layout, System};

static mut DLMALLOC: dlmalloc::Dlmalloc = dlmalloc::DLMALLOC_INIT;

#[stable(feature = "alloc_system_type", since = "1.28.0")]
unsafe impl GlobalAlloc for System {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _lock = lock::lock();
        DLMALLOC.malloc(layout.size(), layout.align())
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let _lock = lock::lock();
        DLMALLOC.calloc(layout.size(), layout.align())
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let _lock = lock::lock();
        DLMALLOC.free(ptr, layout.size(), layout.align())
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _lock = lock::lock();
        DLMALLOC.realloc(ptr, layout.size(), layout.align(), new_size)
    }
}

#[cfg(target_feature = "atomics")]
mod lock {
    use crate::arch::wasm32;
    use crate::sync::atomic::{AtomicI32, Ordering::SeqCst};

    static LOCKED: AtomicI32 = AtomicI32::new(0);

    pub struct DropLock;

    pub fn lock() -> DropLock {
        loop {
            if LOCKED.swap(1, SeqCst) == 0 {
                return DropLock
            }
            unsafe {
                let r = wasm32::i32_atomic_wait(
                    &LOCKED as *const AtomicI32 as *mut i32,
                    1,  // expected value
                    -1, // timeout
                );
                debug_assert!(r == 0 || r == 1);
            }
        }
    }

    impl Drop for DropLock {
        fn drop(&mut self) {
            let r = LOCKED.swap(0, SeqCst);
            debug_assert_eq!(r, 1);
            unsafe {
                wasm32::atomic_notify(
                    &LOCKED as *const AtomicI32 as *mut i32,
                    1, // only one thread
                );
            }
        }
    }
}

#[cfg(not(target_feature = "atomics"))]
mod lock {
    #[inline]
    pub fn lock() {} // no atomics, no threads, that's easy!
}
