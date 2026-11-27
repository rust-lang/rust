// rustfmt-indent_style: Visual
struct Functions {
    RunListenServer: unsafe extern "C" fn(*mut c_void,
     *mut c_char,
     *mut c_char,
     *mut c_char,
     *mut c_void,
     *mut c_void) -> c_int,
}
