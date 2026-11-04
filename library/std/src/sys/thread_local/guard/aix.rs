//! AIX registers thread destructors using `__pt_atexit_np`.
//! Calling it registers a callback that runs on thread termination
//! and upon process exit (for the main thread).

use crate::cell::Cell;
use crate::sys::thread_local::destructors;

pub fn enable() {
    #[thread_local]
    static REGISTERED: Cell<bool> = Cell::new(false);

    unsafe extern "C" {
        fn __pt_atexit_np(flags: i32, dtor: unsafe extern "C" fn(i32) -> i32) -> i32;
    }

    if !REGISTERED.replace(true) {
        let ret = unsafe { __pt_atexit_np(0, run_dtors) };
        if ret != 0 {
            rtabort!("failed to register thread destructor");
        }
    }

    unsafe extern "C" fn run_dtors(_: i32) -> i32 {
        unsafe {
            destructors::run();
            crate::rt::thread_cleanup();
        }
        0
    }
}
