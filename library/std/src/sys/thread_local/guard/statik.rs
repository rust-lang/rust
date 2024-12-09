//! The platform has no threads, so we just need to register
//! a process exit callback.

use crate::cell::Cell;
use crate::sys::thread_local::exit::at_process_exit;
use crate::sys::thread_local::statik::run_dtors;

pub fn enable() {
    struct Registered(Cell<bool>);
    // SAFETY: the target doesn't have threads.
    unsafe impl Sync for Registered {}

    static REGISTERED: Registered = Registered(Cell::new(false));

    if !REGISTERED.0.get() {
        REGISTERED.0.set(true);
        unsafe { at_process_exit(run_process) };
    }

    unsafe extern "C" fn run_process() {
        unsafe { run_dtors() };
        crate::rt::thread_cleanup();
    }
}
