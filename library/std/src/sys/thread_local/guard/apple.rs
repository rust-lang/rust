//! Apple platforms allows registering destructors through [`_tlv_atexit`], which
//! is the mechanism used by the C++ compiler to register destructors of C++
//! static with thread storage duration.
//!
//! Notably, that is the *only* way of registering destructors for native thread-
//! local storage. libdyld [uses pthread TLS keys itself][dyld-tls] to keep track
//! of the memory used for linker variables, so the TLS storage is already freed
//! at the time the destructors of user TLS keys are run. dyld will reinitialize
//! the storage with its initial values, but that still invalidates the pointers
//! stored in the destructor list.
//!
//! It is also worth mention that old versions of libdyld (before dyld-732.8 in
//! macOS 10.15) had a bug that meant registering new TLS destructors from inside
//! a destructor triggered undefined behaviour. This is increasingly irrelevant,
//! but by making sure not to call `_tlv_atexit` when `run` is already registered
//! we can work around this bug in most cases. Initializing a Rust TLS variable
//! from the destructor of a C++ object with thread storage duration will still
//! trigger it, however.
//!
//! [`_tlv_atexit`]: https://github.com/apple-oss-distributions/dyld/blob/fd8d0c4d52320ebf64db34f3cb280310d905c5ae/include/mach-o/dyld.h#L103-L107
//! [dyld-tls]: https://github.com/apple-oss-distributions/dyld/blob/fd8d0c4d52320ebf64db34f3cb280310d905c5ae/libdyld/ThreadLocalVariables.cpp#L385-L386

use crate::cell::Cell;
use crate::ptr;
use crate::sys::thread_local::native::destructors;

pub fn enable() {
    #[thread_local]
    static REGISTERED: Cell<bool> = Cell::new(false);

    unsafe extern "C" {
        fn _tlv_atexit(dtor: unsafe extern "C" fn(*mut u8), arg: *mut u8);
    }

    if !REGISTERED.replace(true) {
        // SAFETY: `run_dtors` is callable (only) at thread exit, the pointer
        // is ignored both by `_tlv_atexit` and `run_dtors`.
        unsafe {
            _tlv_atexit(run_dtors, ptr::null_mut());
        }
    }

    unsafe extern "C" fn run_dtors(_: *mut u8) {
        unsafe {
            destructors::run();
            crate::rt::thread_cleanup();
        }
    }
}
