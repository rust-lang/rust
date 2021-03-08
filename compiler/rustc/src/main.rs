fn main() {
    // Pull in mimalloc when enabled.
    //
    // Note that we're pulling in a static copy of mimalloc which means that to
    // pull it in we need to actually reference its symbols for it to get
    // linked. The two crates we link to here, std and rustc_driver, are both
    // dynamic libraries. That means to pull in mimalloc we actually need to
    // reference allocation symbols one way or another (as this file is the only
    // object code in the rustc executable).
    #[cfg(feature = "libmimalloc-sys")]
    {
        use std::os::raw::{c_int, c_void};

        #[used]
        static _F1: unsafe extern "C" fn(usize, usize) -> *mut c_void = libmimalloc_sys::mi_calloc;
        #[used]
        static _F2: unsafe extern "C" fn(*mut *mut c_void, usize, usize) -> c_int =
            libmimalloc_sys::mi_posix_memalign;
        #[used]
        static _F3: unsafe extern "C" fn(usize, usize) -> *mut c_void =
            libmimalloc_sys::mi_aligned_alloc;
        #[used]
        static _F4: unsafe extern "C" fn(usize) -> *mut c_void = libmimalloc_sys::mi_malloc;
        #[used]
        static _F5: unsafe extern "C" fn(*mut c_void, usize) -> *mut c_void =
            libmimalloc_sys::mi_realloc;
        #[used]
        static _F6: unsafe extern "C" fn(*mut c_void) = libmimalloc_sys::mi_free;
    }

    rustc_driver::set_sigpipe_handler();
    rustc_driver::main()
}
