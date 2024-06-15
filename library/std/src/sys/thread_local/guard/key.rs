//! A lot of UNIX platforms don't have a way to register TLS destructors.
//! Instead, we use one TLS key to register a callback which will run
//! iterate through the destructor list.

use crate::ptr;
use crate::sys::thread_local::destructors;
use crate::sys::thread_local::key::StaticKey;

pub fn enable() {
    static DTORS: StaticKey = StaticKey::new(Some(run));

    // Setting the key value to something other than NULL will result in the
    // destructor being run at thread exit.
    unsafe {
        DTORS.set(ptr::without_provenance_mut(1));
    }

    unsafe extern "C" fn run(_: *mut u8) {
        unsafe {
            destructors::run();
        }
    }
}
