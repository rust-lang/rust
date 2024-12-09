//! A lot of UNIX platforms don't have a specialized way to register TLS
//! destructors for native TLS. Instead, we use one TLS key with a destructor
//! that will run all native TLS destructors in the destructor list.

use crate::ptr;
use crate::sync::atomic::{AtomicBool, Ordering};
use crate::sys::thread_local::exit::at_process_exit;
use crate::sys::thread_local::key::{LazyKey, set};

#[cfg(target_thread_local)]
pub fn enable() {
    fn enable_thread() {
        static DTORS: LazyKey = LazyKey::new(Some(run_thread));

        // Setting the key value to something other than NULL will result in the
        // destructor being run at thread exit.
        unsafe {
            set(DTORS.force(), ptr::without_provenance_mut(1));
        }

        unsafe extern "C" fn run_thread(_: *mut u8) {
            run()
        }
    }

    fn enable_process() {
        static REGISTERED: AtomicBool = AtomicBool::new(false);
        if !REGISTERED.swap(true, Ordering::Relaxed) {
            unsafe { at_process_exit(run_process) };
        }

        unsafe extern "C" fn run_process() {
            run()
        }
    }

    fn run() {
        use crate::sys::thread_local::destructors;

        unsafe {
            destructors::run();
            // On platforms with `__cxa_thread_atexit_impl`, `destructors::run`
            // does nothing on newer systems as the TLS destructors are
            // registered with the system. But because all of those platforms
            // call the destructors of TLS keys after the registered ones, this
            // function will still be run last (at the time of writing).
            crate::rt::thread_cleanup();
        }
    }

    enable_thread();
    enable_process();
}

/// On platforms with key-based TLS, the system runs the destructors for us.
/// We still have to make sure that [`crate::rt::thread_cleanup`] is called,
/// however. This is done by defering the execution of a TLS destructor to
/// the next round of destruction inside the TLS destructors.
///
/// POSIX systems do not run TLS destructors at process exit.
/// Thus we register our own callback to invoke them in that case.
#[cfg(not(target_thread_local))]
pub fn enable() {
    fn enable_thread() {
        const DEFER: *mut u8 = ptr::without_provenance_mut(1);
        const RUN: *mut u8 = ptr::without_provenance_mut(2);

        static CLEANUP: LazyKey = LazyKey::new(Some(run_thread));

        unsafe { set(CLEANUP.force(), DEFER) }

        unsafe extern "C" fn run_thread(state: *mut u8) {
            if state == DEFER {
                // Make sure that this function is run again in the next round of
                // TLS destruction. If there is no futher round, there will be leaks,
                // but that's okay, `thread_cleanup` is not guaranteed to be called.
                unsafe { set(CLEANUP.force(), RUN) }
            } else {
                debug_assert_eq!(state, RUN);
                // If the state is still RUN in the next round of TLS destruction,
                // it means that no other TLS destructors defined by this runtime
                // have been run, as they would have set the state to DEFER.
                crate::rt::thread_cleanup();
            }
        }
    }

    fn enable_process() {
        static REGISTERED: AtomicBool = AtomicBool::new(false);
        if !REGISTERED.swap(true, Ordering::Relaxed) {
            unsafe { at_process_exit(run_process) };
        }

        unsafe extern "C" fn run_process() {
            unsafe { crate::sys::thread_local::key::run_dtors() };
        }
    }

    enable_thread();
    enable_process();
}
