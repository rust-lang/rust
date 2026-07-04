//@compile-flags: -Zmiri-ignore-leaks
#![feature(alloc_error_hook)]

use std::alloc::{GlobalAlloc, Layout, System, set_alloc_error_hook};
use std::ffi::CString;
use std::panic::{self, AssertUnwindSafe, catch_unwind};
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

// `<CStr as ToOwned>::clone_into` must leave the target `CString` valid even when
// the reallocation for its grow path unwinds. See rust-lang/rust#155707.
fn main() {
    set_alloc_error_hook(|_| panic!("alloc error"));
    panic::set_hook(Box::new(|_| {})); // keep the caught panic quiet

    let src = CString::new("a fairly long value").unwrap();
    let mut target = CString::new("x").unwrap();

    ARMED.store(true, Ordering::SeqCst);
    let res = catch_unwind(AssertUnwindSafe(|| src.as_c_str().clone_into(&mut target)));
    ARMED.store(false, Ordering::SeqCst);

    assert!(res.is_err(), "clone_into should have unwound on the alloc failure");
    // `target` must still end in its nul terminator; before the fix it was left empty.
    assert_eq!(target.as_bytes_with_nul().last(), Some(&0));
}
