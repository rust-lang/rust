//! Ensures that thread-local destructors are run on thread exit.

#![cfg(target_thread_local)]
#![unstable(feature = "thread_local_internals", issue = "none")]

use super::{abi, itron::task};
use crate::cell::Cell;
use crate::sys::common::thread_local::run_dtors;

#[thread_local]
static REGISTERED: Cell<bool> = Cell::new(false);

pub fn activate() {
    if !REGISTERED.get() {
        let tid = task::current_task_id_aborting();
        // Register `tls_dtor` to make sure the TLS destructors are called
        // for tasks created by other means than `std::thread`
        unsafe { abi::SOLID_TLS_AddDestructor(tid as i32, run_dtors) };
        REGISTERED.set(true);
    }
}
