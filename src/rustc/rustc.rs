fn main() {
    // Pull in mimalloc when enabled.
    //
    // Note that we're pulling in a static copy of mimalloc which means that to
    // pull it in we need to actually reference its symbols for it to get
    // linked. The two crates we link to here, std and rustc_driver, are both
    // dynamic libraries. That means to pull in mimalloc we need to actually
    // reference allocation symbols one way or another (as this file is the only
    // object code in the rustc executable).
    #[cfg(feature = "mimalloc-sys")]
    {
        use std::os::raw::{c_void, c_int};

        #[used]
        static _F1: unsafe extern fn(usize, usize) -> *mut c_void =
            mimalloc_sys::mi_calloc;
        // NOTE: this symbol calls mi_malloc_aligned
        // #[used]
        // static _F2: unsafe extern fn(*mut *mut c_void, usize, usize) -> c_int =
        //   mimalloc_sys::mi_posix_memalign;
        #[used]
        static _F3: unsafe extern fn(usize, usize) -> *mut c_void =
            mimalloc_sys::mi_malloc_aligned;
        #[used]
        static _F4: unsafe extern fn(usize) -> *mut c_void =
            mimalloc_sys::mi_malloc;
        #[used]
        static _F5: unsafe extern fn(*mut c_void, usize) -> *mut c_void =
            mimalloc_sys::mi_realloc;
        #[used]
        static _F6: unsafe extern fn(*mut c_void) =
            mimalloc_sys::mi_free;
    }

    rustc_driver::set_sigpipe_handler();
    rustc_driver::main()
}
