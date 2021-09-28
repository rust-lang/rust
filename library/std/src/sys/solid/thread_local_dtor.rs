#![cfg(target_thread_local)]
#![unstable(feature = "thread_local_internals", issue = "none")]

// Simplify dtor registration by using a list of destructors.

use super::{abi, itron::task};
use crate::cell::Cell;
use crate::ptr;

#[thread_local]
static DTORS: Cell<*mut List> = Cell::new(ptr::null_mut());

type List = Vec<(*mut u8, unsafe extern "C" fn(*mut u8))>;

pub unsafe fn register_dtor(t: *mut u8, dtor: unsafe extern "C" fn(*mut u8)) {
    if DTORS.get().is_null() {
        let tid = task::current_task_id_aborting();
        let v: Box<List> = box Vec::new();
        DTORS.set(Box::into_raw(v));

        // Register `tls_dtor` to make sure the TLS destructors are called
        // for tasks created by other means than `std::thread`
        unsafe { abi::SOLID_TLS_AddDestructor(tid as i32, tls_dtor) };
    }

    let list: &mut List = unsafe { &mut *DTORS.get() };
    list.push((t, dtor));
}

pub unsafe fn run_dtors() {
    let ptr = DTORS.get();
    if !ptr.is_null() {
        // Swap the destructor list, call all registered destructors,
        // and repeat this until the list becomes permanently empty.
        while let Some(list) = Some(crate::mem::replace(unsafe { &mut *ptr }, Vec::new()))
            .filter(|list| !list.is_empty())
        {
            for (ptr, dtor) in list.into_iter() {
                unsafe { dtor(ptr) };
            }
        }

        // Drop the destructor list
        unsafe { Box::from_raw(DTORS.replace(ptr::null_mut())) };
    }
}

unsafe extern "C" fn tls_dtor(_unused: *mut u8) {
    unsafe { run_dtors() };
}
