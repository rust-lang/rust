fn main() {
    // Pull in jemalloc when enabled.
    //
    // Note that we're pulling in a static copy of jemalloc which means that to
    // pull it in we need to actually reference its symbols for it to get
    // linked. The two crates we link to here, std and rustc_driver, are both
    // dynamic libraries. That means to pull in jemalloc we need to actually
    // reference allocation symbols one way or another (as this file is the only
    // object code in the rustc executable).
    #[cfg(feature = "jemalloc-sys")]
    {
        use std::os::raw::{c_void, c_int};

        #[used]
        static _F1: unsafe extern fn(usize, usize) -> *mut c_void =
            jemalloc_sys::calloc;
        #[used]
        static _F2: unsafe extern fn(*mut *mut c_void, usize, usize) -> c_int =
            jemalloc_sys::posix_memalign;
        #[used]
        static _F3: unsafe extern fn(usize, usize) -> *mut c_void =
            jemalloc_sys::aligned_alloc;
        #[used]
        static _F4: unsafe extern fn(usize) -> *mut c_void =
            jemalloc_sys::malloc;
        #[used]
        static _F5: unsafe extern fn(*mut c_void, usize) -> *mut c_void =
            jemalloc_sys::realloc;
        #[used]
        static _F6: unsafe extern fn(*mut c_void) =
            jemalloc_sys::free;
    }

    rustc_driver::set_sigpipe_handler();
    rustc_driver::main()
}
