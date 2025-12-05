//! SOLID, just like macOS, has an API to register TLS destructors. But since
//! it does not allow specifying an argument to that function, and will not run
//! destructors for terminated tasks, we still keep our own list.

use crate::cell::Cell;
use crate::sys::pal::abi;
use crate::sys::pal::itron::task;
use crate::sys::thread_local::destructors;

pub fn enable() {
    #[thread_local]
    static REGISTERED: Cell<bool> = Cell::new(false);

    if !REGISTERED.replace(true) {
        let tid = task::current_task_id_aborting();
        // Register `tls_dtor` to make sure the TLS destructors are called
        // for tasks created by other means than `std::thread`
        unsafe { abi::SOLID_TLS_AddDestructor(tid as i32, tls_dtor) };
    }

    unsafe extern "C" fn tls_dtor(_unused: *mut u8) {
        unsafe {
            destructors::run();
            crate::rt::thread_cleanup();
        }
    }
}
