//! Support for Windows TLS destructors.
//!
//! Windows has an API to provide a destructor for a FLS (fiber local storage) variable,
//! which behaves similarly to a TLS variable for our purpose [1].
//!
//! All TLS destructors are tracked by *us*, not the Windows runtime.
//! This means that we have a global list of destructors for
//! each TLS key or variable that we know about.
//!
//! [1]: https://devblogs.microsoft.com/oldnewthing/20191011-00/?p=102989

use core::ffi::c_void;

use crate::cell::Cell;
use crate::ptr;
use crate::sys::c;

pub type Key = u32;

unsafe fn create(dtor: c::PFLS_CALLBACK_FUNCTION) -> Key {
    let key_result = unsafe { c::FlsAlloc(dtor) };

    if key_result == c::FLS_OUT_OF_INDEXES {
        rtabort!("out of FLS keys");
    }

    key_result
}

unsafe fn set(key: Key, ptr: *const c_void) {
    let result = unsafe { c::FlsSetValue(key, ptr) };

    if result == c::FALSE {
        rtabort!("failed to set FLS value");
    }
}

pub fn enable() {
    #[thread_local]
    static REGISTERED: Cell<bool> = Cell::new(false);

    if !REGISTERED.replace(true) {
        unsafe {
            let key = create(Some(cleanup));
            set(key, ptr::dangling());
        };
    }
}

unsafe extern "system" fn cleanup(_ptr: *const c_void) {
    unsafe {
        #[cfg(target_thread_local)]
        super::super::destructors::run();
        #[cfg(not(target_thread_local))]
        super::super::key::run_dtors();
    }

    crate::rt::thread_cleanup();
}
