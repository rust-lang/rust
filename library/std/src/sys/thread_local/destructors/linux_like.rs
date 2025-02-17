//! Destructor registration for Linux-like systems.
//!
//! Since what appears to be version 2.18, glibc has shipped the
//! `__cxa_thread_atexit_impl` symbol which GCC and clang both use to invoke
//! destructors in C++ thread_local globals. This function does exactly what
//! we want: it schedules a callback which will be run at thread exit with the
//! provided argument.
//!
//! Unfortunately, our minimum supported glibc version (at the time of writing)
//! is 2.17, so we can only link this symbol weakly and need to use the
//! [`list`](super::list) destructor implementation as fallback.

use crate::mem::transmute;

// FIXME: The Rust compiler currently omits weakly function definitions (i.e.,
// __cxa_thread_atexit_impl) and its metadata from LLVM IR.
#[no_sanitize(cfi, kcfi)]
pub unsafe fn register(t: *mut u8, dtor: unsafe extern "C" fn(*mut u8)) {
    /// This is necessary because the __cxa_thread_atexit_impl implementation
    /// std links to by default may be a C or C++ implementation that was not
    /// compiled using the Clang integer normalization option.
    #[cfg(sanitizer_cfi_normalize_integers)]
    use core::ffi::c_int;
    #[cfg(not(sanitizer_cfi_normalize_integers))]
    #[cfi_encoding = "i"]
    #[repr(transparent)]
    #[allow(non_camel_case_types)]
    pub struct c_int(#[allow(dead_code)] pub core::ffi::c_int);

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

    if let Some(f) = unsafe { __cxa_thread_atexit_impl } {
        unsafe {
            f(
                transmute::<unsafe extern "C" fn(*mut u8), unsafe extern "C" fn(*mut libc::c_void)>(
                    dtor,
                ),
                t.cast(),
                (&raw const __dso_handle) as *mut _,
            );
        }
    } else {
        unsafe {
            super::list::register(t, dtor);
        }
    }
}
