//! Implements thread-local destructors that are not associated with any
//! particular data.

#![unstable(feature = "thread_local_internals", issue = "none")]
#![cfg(target_thread_local)]

pub unsafe fn register_dtor(t: *mut u8, dtor: unsafe extern "C" fn(*mut u8)) {
    crate::sys::thread_local_key::register_keyless_dtor(t, dtor)
}
