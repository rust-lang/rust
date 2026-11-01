//! Regression test for the panic-safety fix in rust-lang/rust#155707.
//!
//! rust-lang/rust#70201 gave `<CStr as ToOwned>::clone_into` a path that moved the
//! target `CString`'s buffer out before growing a `Vec`; if that growth's allocation
//! failed and unwound, the target was left without its nul terminator. This only
//! reproduces under Miri: in a normal build the `#[global_allocator]` below can't
//! intercept the reallocation inside `CString::clone_into` (it lives in libstd, which
//! library tests link with `-C prefer-dynamic`), so as a regular test it just checks
//! the happy path.

// Disabled under Miri on Windows: a `#[global_allocator]` wrapping `System` trips
// Stacked Borrows there, and it affects libtest's own allocations, not just this test
// (so `#[ignore]` would not be enough). See <https://github.com/rust-lang/miri/issues/2104>.
#![cfg(not(all(miri, windows)))]
#![feature(alloc_error_hook)]

use std::alloc::{GlobalAlloc, Layout, System, set_alloc_error_hook};
use std::ffi::CString;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};

// Once armed, the first allocation of 8 bytes or more fails and disarms, so the
// reallocation inside `clone_into`'s grow path fails while the runtime's own
// smaller allocations keep succeeding.
struct OneShotFailingAlloc;

static ARMED: AtomicBool = AtomicBool::new(false);

unsafe impl GlobalAlloc for OneShotFailingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() >= 8 && ARMED.swap(false, Ordering::SeqCst) {
            return core::ptr::null_mut();
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size >= 8 && ARMED.swap(false, Ordering::SeqCst) {
            return core::ptr::null_mut();
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: OneShotFailingAlloc = OneShotFailingAlloc;

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn clone_into_alloc_failure_leaves_target_valid() {
    set_alloc_error_hook(|_| panic!("alloc error"));

    let src = CString::new("a fairly long value").unwrap();
    let mut target = CString::new("x").unwrap();

    ARMED.store(true, Ordering::SeqCst);
    // Under Miri the failing allocator is honored, so this reallocation unwinds; in a
    // normal build the allocator can't intercept it and `clone_into` just succeeds.
    let res = catch_unwind(AssertUnwindSafe(|| src.as_c_str().clone_into(&mut target)));
    ARMED.store(false, Ordering::SeqCst);

    if cfg!(miri) {
        assert!(res.is_err(), "clone_into should have unwound on the alloc failure");
    }
    // Either way `target` must still end in its nul terminator. Before the fix the Miri
    // unwind left it empty (also caught as a bad write in `CString`'s destructor).
    assert_eq!(target.as_bytes_with_nul().last(), Some(&0));
}
