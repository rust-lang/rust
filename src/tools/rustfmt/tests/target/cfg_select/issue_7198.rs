//@ rustfmt-show_parse_errors: false
// There are parse errors that we don't wont to show in test output

cfg_select! {
    any() => {
        extern "C" {
            fn __cxa_thread_atexit_impl(
                unsafe extern "C" fn(*mut libc::c_void),
                *mut libc::c_void,
                *mut libc::c_void,
            ) -> c_int;
        }
    }
}
