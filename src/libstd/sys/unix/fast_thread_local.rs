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
#[cfg(target_os = "macos")]
pub unsafe fn register_dtor(t: *mut u8, dtor: unsafe extern fn(*mut u8)) {
    use crate::cell::Cell;
    use crate::ptr;

    #[thread_local]
    static REGISTERED: Cell<bool> = Cell::new(false);
    if !REGISTERED.get() {
        _tlv_atexit(run_dtors, ptr::null_mut());
        REGISTERED.set(true);
    }

    type List = Vec<(*mut u8, unsafe extern fn(*mut u8))>;

    #[thread_local]
    static DTORS: Cell<*mut List> = Cell::new(ptr::null_mut());
    if DTORS.get().is_null() {
        let v: Box<List> = box Vec::new();
        DTORS.set(Box::into_raw(v));
    }

    extern {
        fn _tlv_atexit(dtor: unsafe extern fn(*mut u8),
                       arg: *mut u8);
    }

    let list: &mut List = &mut *DTORS.get();
    list.push((t, dtor));

    unsafe extern fn run_dtors(_: *mut u8) {
        let mut ptr = DTORS.replace(ptr::null_mut());
        while !ptr.is_null() {
            let list = Box::from_raw(ptr);
            for (ptr, dtor) in list.into_iter() {
                dtor(ptr);
            }
            ptr = DTORS.replace(ptr::null_mut());
        }
    }
}
