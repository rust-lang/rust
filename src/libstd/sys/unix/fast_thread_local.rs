#![cfg(target_thread_local)]
#![unstable(feature = "thread_local_internals", issue = "0")]

// Since what appears to be glibc 2.18 this symbol has been shipped which
// GCC and clang both use to invoke destructors in thread_local globals, so
// let's do the same!
//
// Note, however, that we run on lots older linuxes, as well as cross
// compiling from a newer linux to an older linux, so we also have a
// fallback implementation to use as well.
//
// Due to rust-lang/rust#18804, make sure this is not generic!
#[cfg(any(target_os = "linux", target_os = "fuchsia", target_os = "hermit"))]
pub unsafe fn register_dtor(t: *mut u8, dtor: unsafe extern fn(*mut u8)) {
    use libc;
    use mem;
    use sys_common::thread_local::register_dtor_fallback;

    extern {
        #[linkage = "extern_weak"]
        static __dso_handle: *mut u8;
        #[linkage = "extern_weak"]
        static __cxa_thread_atexit_impl: *const libc::c_void;
    }
    if !__cxa_thread_atexit_impl.is_null() {
        type F = unsafe extern fn(dtor: unsafe extern fn(*mut u8),
                                  arg: *mut u8,
                                  dso_handle: *mut u8) -> libc::c_int;
        mem::transmute::<*const libc::c_void, F>(__cxa_thread_atexit_impl)
            (dtor, t, &__dso_handle as *const _ as *mut _);
        return
    }
    register_dtor_fallback(t, dtor);
}

// macOS's analog of the above linux function is this _tlv_atexit function.
// The disassembly of thread_local globals in C++ (at least produced by
// clang) will have this show up in the output.
#[cfg(target_os = "macos")]
pub unsafe fn register_dtor(t: *mut u8, dtor: unsafe extern fn(*mut u8)) {
    extern {
        fn _tlv_atexit(dtor: unsafe extern fn(*mut u8),
                       arg: *mut u8);
    }
    _tlv_atexit(dtor, t);
}

pub fn requires_move_before_drop() -> bool {
    // The macOS implementation of TLS apparently had an odd aspect to it
    // where the pointer we have may be overwritten while this destructor
    // is running. Specifically if a TLS destructor re-accesses TLS it may
    // trigger a re-initialization of all TLS variables, paving over at
    // least some destroyed ones with initial values.
    //
    // This means that if we drop a TLS value in place on macOS that we could
    // revert the value to its original state halfway through the
    // destructor, which would be bad!
    //
    // Hence, we use `ptr::read` on macOS (to move to a "safe" location)
    // instead of drop_in_place.
    cfg!(target_os = "macos")
}
