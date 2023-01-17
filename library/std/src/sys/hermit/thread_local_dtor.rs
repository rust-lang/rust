#![cfg(target_thread_local)]
#![unstable(feature = "thread_local_internals", issue = "none")]

// Simplify dtor registration by using a list of destructors.
// The this solution works like the implementation of macOS and
// doesn't additional OS support

use crate::mem;

#[thread_local]
static mut DTORS: Vec<(*mut u8, unsafe extern "C" fn(*mut u8))> = Vec::new();

pub unsafe fn register_dtor(t: *mut u8, dtor: unsafe extern "C" fn(*mut u8)) {
    let list = &mut DTORS;
    list.push((t, dtor));
}

// every thread call this function to run through all possible destructors
pub unsafe fn run_dtors() {
    let mut list = mem::take(&mut DTORS);
    while !list.is_empty() {
        for (ptr, dtor) in list {
            dtor(ptr);
        }
        list = mem::take(&mut DTORS);
    }
}
