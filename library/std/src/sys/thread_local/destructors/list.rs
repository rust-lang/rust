use crate::alloc::System;
use crate::cell::RefCell;
use crate::sync::atomic::{Atomic, Ordering};
use crate::sys::thread_local::guard;

#[thread_local]
static DTORS: RefCell<Vec<(*mut u8, unsafe extern "C" fn(*mut u8)), System>> =
    RefCell::new(Vec::new_in(System));

static DISABLE_DTORS: Atomic<bool> = Atomic::<bool>::new(false);

/// Permanently disable running TLS dtors.
pub fn disable() {
    DISABLE_DTORS.store(true, Ordering::Release);
}

pub unsafe fn register(t: *mut u8, dtor: unsafe extern "C" fn(*mut u8)) {
    let Ok(mut dtors) = DTORS.try_borrow_mut() else {
        rtabort!("the System allocator may not use TLS with destructors")
    };
    guard::enable();

    // Avoid calling the alloc error hook
    if dtors.capacity() == dtors.len() {
        dtors.try_reserve(1).unwrap_or_else(|_| rtabort!("Failed to grow TLS destructor list"))
    }
    dtors.push((t, dtor));
}

/// The [`guard`] module contains platform-specific functions which will run this
/// function on thread exit if [`guard::enable`] has been called.
///
/// # Safety
///
/// May only be run on thread exit to guarantee that there are no live references
/// to TLS variables while they are destroyed.
pub unsafe fn run() {
    if DISABLE_DTORS.load(Ordering::Acquire) {
        if let Ok(mut dtors) = DTORS.try_borrow_mut() {
            *dtors = Vec::new_in(System);
        }
        return;
    }
    loop {
        let mut dtors = DTORS.borrow_mut();
        match dtors.pop() {
            Some((t, dtor)) => {
                drop(dtors);
                unsafe {
                    dtor(t);
                }
            }
            None => {
                // Free the list memory.
                *dtors = Vec::new_in(System);
                break;
            }
        }
    }
}
