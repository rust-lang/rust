#![cfg(target_thread_local)]
#![unstable(feature = "thread_local_internals", issue = "none")]

// Simplify dtor registration by using a list of destructors.

use super::{abi, itron::task};
use crate::cell::Cell;
use crate::mem;

#[thread_local]
static REGISTERED: Cell<bool> = Cell::new(false);

#[thread_local]
static mut DTORS: Vec<(*mut u8, unsafe extern "C" fn(*mut u8))> = Vec::new();

pub unsafe fn register_dtor(t: *mut u8, dtor: unsafe extern "C" fn(*mut u8)) {
    if !REGISTERED.get() {
        let tid = task::current_task_id_aborting();
        // Register `tls_dtor` to make sure the TLS destructors are called
        // for tasks created by other means than `std::thread`
        unsafe { abi::SOLID_TLS_AddDestructor(tid as i32, tls_dtor) };
        REGISTERED.set(true);
    }

    let list = unsafe { &mut DTORS };
    list.push((t, dtor));
}

pub unsafe fn run_dtors() {
    let mut list = mem::take(unsafe { &mut DTORS });
    while !list.is_empty() {
        for (ptr, dtor) in list {
            unsafe { dtor(ptr) };
        }

        list = mem::take(unsafe { &mut DTORS });
    }
}

unsafe extern "C" fn tls_dtor(_unused: *mut u8) {
    unsafe { run_dtors() };
}
