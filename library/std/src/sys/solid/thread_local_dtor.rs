#![cfg(target_thread_local)]
#![unstable(feature = "thread_local_internals", issue = "none")]

// Simplify dtor registration by using a list of destructors.

use super::{abi, itron::task};
use crate::cell::{Cell, RefCell};

#[thread_local]
static REGISTERED: Cell<bool> = Cell::new(false);

#[thread_local]
static DTORS: RefCell<Vec<(*mut u8, unsafe extern "C" fn(*mut u8))>> = RefCell::new(Vec::new());

pub unsafe fn register_dtor(t: *mut u8, dtor: unsafe extern "C" fn(*mut u8)) {
    if !REGISTERED.get() {
        let tid = task::current_task_id_aborting();
        // Register `tls_dtor` to make sure the TLS destructors are called
        // for tasks created by other means than `std::thread`
        unsafe { abi::SOLID_TLS_AddDestructor(tid as i32, tls_dtor) };
        REGISTERED.set(true);
    }

    match DTORS.try_borrow_mut() {
        Ok(mut dtors) => dtors.push((t, dtor)),
        Err(_) => rtabort!("global allocator may not use TLS"),
    }
}

pub unsafe fn run_dtors() {
    let mut list = DTORS.take();
    while !list.is_empty() {
        for (ptr, dtor) in list {
            unsafe { dtor(ptr) };
        }

        list = DTORS.take();
    }
}

unsafe extern "C" fn tls_dtor(_unused: *mut u8) {
    unsafe { run_dtors() };
}
