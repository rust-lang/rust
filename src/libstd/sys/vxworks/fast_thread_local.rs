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
pub unsafe fn register_dtor(t: *mut u8, dtor: unsafe extern fn(*mut u8)) {
    use crate::mem;
    use crate::sys_common::thread_local::register_dtor_fallback;

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

pub fn requires_move_before_drop() -> bool {
    false
}
