//! A lot of UNIX platforms don't have a specialized way to register TLS
//! destructors for native TLS. Instead, we use one TLS key with a destructor
//! that will run all native TLS destructors in the destructor list.

use crate::ptr;
use crate::sys::thread_local::key::{LazyKey, set};

pub fn enable() {
    use crate::sys::thread_local::destructors;

    static DTORS: LazyKey = LazyKey::new(Some(run));

    // Setting the key value to something other than NULL will result in the
    // destructor being run at thread exit.
    unsafe {
        set(DTORS.force(), ptr::without_provenance_mut(1));
    }

    unsafe extern "C" fn run(_: *mut u8) {
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
}
