//! Ensures that thread-local destructors are run on thread exit.

#![cfg(target_thread_local)]
#![unstable(feature = "thread_local_internals", issue = "none")]

use crate::ptr;
use crate::sys::common::thread_local::run_dtors;

// Since what appears to be glibc 2.18 this symbol has been shipped which
// GCC and clang both use to invoke destructors in thread_local globals, so
// let's do the same!
//
// Note, however, that we run on lots older linuxes, as well as cross
// compiling from a newer linux to an older linux, so we also have a
// fallback implementation to use as well.
#[cfg_attr(bootstrap, allow(unexpected_cfgs))]
#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "fuchsia",
    target_os = "redox",
    target_os = "hurd",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "dragonfly"
))]
// FIXME: The Rust compiler currently omits weakly function definitions (i.e.,
// __cxa_thread_atexit_impl) and its metadata from LLVM IR.
#[no_sanitize(cfi, kcfi)]
pub fn activate() {
    use crate::cell::Cell;
    use crate::mem;
    use crate::sys_common::thread_local_key::StaticKey;

    /// This is necessary because the __cxa_thread_atexit_impl implementation
    /// std links to by default may be a C or C++ implementation that was not
    /// compiled using the Clang integer normalization option.
    #[cfg(sanitizer_cfi_normalize_integers)]
    use core::ffi::c_int;
    #[cfg(not(sanitizer_cfi_normalize_integers))]
    #[cfi_encoding = "i"]
    #[repr(transparent)]
    pub struct c_int(pub libc::c_int);

    extern "C" {
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

    unsafe {
        if let Some(atexit) = __cxa_thread_atexit_impl {
            #[thread_local]
            static REGISTERED: Cell<bool> = Cell::new(false);
            if !REGISTERED.get() {
                atexit(
                    mem::transmute::<
                        unsafe extern "C" fn(*mut u8),
                        unsafe extern "C" fn(*mut libc::c_void),
                    >(run_dtors),
                    ptr::null_mut(),
                    &__dso_handle as *const _ as *mut _,
                );
                REGISTERED.set(true);
            }
        } else {
            static KEY: StaticKey = StaticKey::new(Some(run_dtors));

            KEY.set(ptr::invalid_mut(1));
        }
    }
}

// We hook into macOS's analog of the above linux function, _tlv_atexit. OSX
// will run `run_dtors` before any TLS slots get freed, and when the main thread
// exits.
#[cfg(any(target_os = "macos", target_os = "ios", target_os = "watchos", target_os = "tvos"))]
pub fn activate() {
    use crate::cell::Cell;

    extern "C" {
        fn _tlv_atexit(dtor: unsafe extern "C" fn(*mut u8), arg: *mut u8);
    }

    #[thread_local]
    static REGISTERED: Cell<bool> = Cell::new(false);

    if !REGISTERED.get() {
        unsafe {
            _tlv_atexit(run_dtors, ptr::null_mut());
            REGISTERED.set(true);
        }
    }
}

#[cfg(any(
    target_os = "vxworks",
    target_os = "horizon",
    target_os = "emscripten",
    target_os = "aix"
))]
#[cfg_attr(target_family = "wasm", allow(unused))] // might remain unused depending on target details (e.g. wasm32-unknown-emscripten)
pub fn activate() {
    use crate::sys_common::thread_local_key::StaticKey;

    static KEY: StaticKey = StaticKey::new(Some(run_dtors));

    unsafe {
        KEY.set(ptr::invalid_mut(1));
    }
}
