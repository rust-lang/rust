//! Destructor registration for Linux-like systems.
//!
//! Since what appears to be version 2.18, glibc has shipped the
//! `__cxa_thread_atexit_impl` symbol which GCC and clang both use to invoke
//! destructors in C++ thread_local globals. This function schedules a callback
//! which will be run at thread exit with the provided argument.
//!
//! Unfortunately, our minimum supported glibc version (at the time of writing)
//! is 2.17, so we can only link this symbol weakly and need to use the pthread-
//! TLS-key-based guard as a fallback.

use crate::cell::Cell;
#[cfg(sanitizer_cfi_normalize_integers)]
use crate::ffi::c_int;
use crate::ptr;
use crate::sys::thread_local::native::destructors;

/// This is necessary because the __cxa_thread_atexit_impl implementation
/// std links to by default may be a C or C++ implementation that was not
/// compiled using the Clang integer normalization option.
#[cfg(not(sanitizer_cfi_normalize_integers))]
#[cfi_encoding = "i"]
#[repr(transparent)]
#[allow(non_camel_case_types)]
struct c_int(#[allow(dead_code)] pub core::ffi::c_int);

unsafe extern "C" {
    #[linkage = "extern_weak"]
    static __dso_handle: *mut u8;
    #[linkage = "extern_weak"]
    static __cxa_thread_atexit_impl: Option<
        extern "C" fn(
            unsafe extern "C" fn(*mut libc::c_void),
            *mut libc::c_void,
            *mut libc::c_void,
        ) -> c_int,
    >;
}

pub fn enable() {
    #[thread_local]
    static REGISTERED: Cell<bool> = Cell::new(false);

    if !REGISTERED.replace(true) {
        if let Some(thread_atexit) = unsafe { __cxa_thread_atexit_impl }
            && let dso = unsafe { __dso_handle }
            && !dso.is_null()
        {
            thread_atexit(run, ptr::null_mut(), dso as *mut _);
        } else {
            super::key::enable();
        }
    }
}

unsafe extern "C" fn run(_: *mut libc::c_void) {
    unsafe {
        destructors::run();
        crate::rt::thread_cleanup();
    }
}
