#![cfg(target_thread_local)]
#![unstable(feature = "thread_local_internals", issue = "none")]

// Simplify dtor registration by using a list of destructors.
// The this solution works like the implementation of macOS and
// doesn't additional OS support

use crate::cell::Cell;
use crate::ptr;

#[thread_local]
static DTORS: Cell<*mut List> = Cell::new(ptr::null_mut());

type List = Vec<(*mut u8, unsafe extern "C" fn(*mut u8))>;

pub unsafe fn register_dtor(t: *mut u8, dtor: unsafe extern "C" fn(*mut u8)) {
    if DTORS.get().is_null() {
        let v: Box<List> = box Vec::new();
        DTORS.set(Box::into_raw(v));
    }

    let list: &mut List = &mut *DTORS.get();
    list.push((t, dtor));
}

// every thread call this function to run through all possible destructors
pub unsafe fn run_dtors() {
    let mut ptr = DTORS.replace(ptr::null_mut());
    while !ptr.is_null() {
        let list = Box::from_raw(ptr);
        for (ptr, dtor) in list.into_iter() {
            dtor(ptr);
        }
        ptr = DTORS.replace(ptr::null_mut());
    }
}
