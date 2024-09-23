//! A lot of UNIX platforms don't have a specialized way to register TLS
//! destructors for native TLS. Instead, we use one TLS key with a destructor
//! that will run all native TLS destructors in the destructor list.

use crate::ptr;
use crate::sys::thread_local::destructors;
use crate::sys::thread_local::key::{LazyKey, set};

pub fn enable() {
    static DTORS: LazyKey = LazyKey::new(Some(run));

    // Setting the key value to something other than NULL will result in the
    // destructor being run at thread exit.
    unsafe {
        set(DTORS.force(), ptr::without_provenance_mut(1));
    }

    unsafe extern "C" fn run(_: *mut u8) {
        unsafe {
            destructors::run();
        }
    }
}
