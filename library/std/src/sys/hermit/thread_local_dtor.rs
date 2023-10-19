#![cfg(target_thread_local)]
#![unstable(feature = "thread_local_internals", issue = "none")]

// Simplify dtor registration by using a list of destructors.
// The this solution works like the implementation of macOS and
// doesn't additional OS support

use crate::cell::RefCell;

#[thread_local]
static DTORS: RefCell<Vec<(*mut u8, unsafe extern "C" fn(*mut u8))>> = RefCell::new(Vec::new());

pub unsafe fn register_dtor(t: *mut u8, dtor: unsafe extern "C" fn(*mut u8)) {
    match DTORS.try_borrow_mut() {
        Ok(mut dtors) => dtors.push((t, dtor)),
        Err(_) => rtabort!("global allocator may not use TLS"),
    }
}

// every thread call this function to run through all possible destructors
pub unsafe fn run_dtors() {
    let mut list = DTORS.take();
    while !list.is_empty() {
        for (ptr, dtor) in list {
            dtor(ptr);
        }
        list = DTORS.take();
    }
}
