#![cfg(target_thread_local)]
#![unstable(feature = "thread_local_internals", issue = "none")]

//! Provides thread-local destructors without an associated "key", which
//! can be more efficient.

// Since what appears to be glibc 2.18 this symbol has been shipped which
// GCC and clang both use to invoke destructors in thread_local globals, so
// let's do the same!
//
// Note, however, that we run on lots older linuxes, as well as cross
// compiling from a newer linux to an older linux, so we also have a
// fallback implementation to use as well.
#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "fuchsia",
    target_os = "redox",
    target_os = "hurd",
    target_os = "netbsd",
    target_os = "dragonfly"
))]
// FIXME: The Rust compiler currently omits weakly function definitions (i.e.,
// __cxa_thread_atexit_impl) and its metadata from LLVM IR.
#[no_sanitize(cfi, kcfi)]
pub unsafe fn register_dtor(t: *mut u8, dtor: unsafe extern "C" fn(*mut u8)) {
    use crate::mem;
    use crate::sys_common::thread_local_dtor::register_dtor_fallback;

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

    if let Some(f) = __cxa_thread_atexit_impl {
        unsafe {
            f(
                mem::transmute::<
                    unsafe extern "C" fn(*mut u8),
                    unsafe extern "C" fn(*mut libc::c_void),
                >(dtor),
                t.cast(),
                &__dso_handle as *const _ as *mut _,
            );
        }
        return;
    }
    register_dtor_fallback(t, dtor);
}

// This implementation is very similar to register_dtor_fallback in
// sys_common/thread_local.rs. The main difference is that we want to hook into
// macOS's analog of the above linux function, _tlv_atexit. OSX will run the
// registered dtors before any TLS slots get freed, and when the main thread
// exits.
//
// Unfortunately, calling _tlv_atexit while tls dtors are running is UB. The
// workaround below is to register, via _tlv_atexit, a custom DTOR list once per
// thread. thread_local dtors are pushed to the DTOR list without calling
// _tlv_atexit.
#[cfg(any(target_os = "macos", target_os = "ios", target_os = "watchos", target_os = "tvos"))]
pub unsafe fn register_dtor(t: *mut u8, dtor: unsafe extern "C" fn(*mut u8)) {
    use crate::cell::{Cell, RefCell};
    use crate::ptr;

    #[thread_local]
    static REGISTERED: Cell<bool> = Cell::new(false);

    #[thread_local]
    static DTORS: RefCell<Vec<(*mut u8, unsafe extern "C" fn(*mut u8))>> = RefCell::new(Vec::new());

    if !REGISTERED.get() {
        _tlv_atexit(run_dtors, ptr::null_mut());
        REGISTERED.set(true);
    }

    extern "C" {
        fn _tlv_atexit(dtor: unsafe extern "C" fn(*mut u8), arg: *mut u8);
    }

    match DTORS.try_borrow_mut() {
        Ok(mut dtors) => dtors.push((t, dtor)),
        Err(_) => rtabort!("global allocator may not use TLS"),
    }

    unsafe extern "C" fn run_dtors(_: *mut u8) {
        let mut list = DTORS.take();
        while !list.is_empty() {
            for (ptr, dtor) in list {
                dtor(ptr);
            }
            list = DTORS.take();
        }
    }
}

#[cfg(any(
    target_os = "vxworks",
    target_os = "horizon",
    target_os = "emscripten",
    target_os = "aix",
    target_os = "freebsd",
))]
#[cfg_attr(target_family = "wasm", allow(unused))] // might remain unused depending on target details (e.g. wasm32-unknown-emscripten)
pub unsafe fn register_dtor(t: *mut u8, dtor: unsafe extern "C" fn(*mut u8)) {
    use crate::sys_common::thread_local_dtor::register_dtor_fallback;
    register_dtor_fallback(t, dtor);
}
