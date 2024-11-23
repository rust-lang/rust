//! wasm32-wasip1 has pthreads support.

use crate::cell::Cell;
use crate::sync::atomic::{AtomicBool, Ordering};
use crate::sys::thread_local::destructors;
use crate::{ffi, ptr};

// Add a few symbols not in upstream `libc` just yet.
mod libc {
    pub use libc::*;

    use crate::ffi;

    #[allow(non_camel_case_types)]
    pub type pthread_key_t = ffi::c_uint;

    extern "C" {
        pub fn pthread_key_create(
            key: *mut pthread_key_t,
            destructor: unsafe extern "C" fn(*mut ffi::c_void),
        ) -> ffi::c_int;

        pub fn pthread_setspecific(key: pthread_key_t, value: *const ffi::c_void) -> ffi::c_int;
    }
}

pub fn enable() {
    enable_main();
    enable_thread();
}

fn enable_main() {
    static REGISTERED: AtomicBool = AtomicBool::new(false);

    if !REGISTERED.swap(true, Ordering::AcqRel) {
        unsafe {
            assert_eq!(libc::atexit(run_main_dtors), 0);
        }
    }

    extern "C" fn run_main_dtors() {
        unsafe {
            destructors::run();
            crate::rt::thread_cleanup();
        }
    }
}

fn enable_thread() {
    #[thread_local]
    static REGISTERED: Cell<bool> = Cell::new(false);

    if !REGISTERED.replace(true) {
        unsafe {
            let mut key: libc::pthread_key_t = 0;
            assert_eq!(libc::pthread_key_create(&mut key, run_thread_dtors), 0);

            // We must set the value to a non-NULL pointer value so that
            // the destructor is run on thread exit. The pointer is only
            // passed to run_dtors and never dereferenced.
            assert_eq!(libc::pthread_setspecific(key, ptr::without_provenance(1)), 0);
        }
    }

    extern "C" fn run_thread_dtors(_: *mut ffi::c_void) {
        unsafe {
            destructors::run();
            crate::rt::thread_cleanup();
        }
    }
}
