// ignore-windows: No libc on Windows

#![feature(rustc_private)]

extern crate libc;

use std::ffi::CString;

fn main() {
    unsafe {
        let thread_name = CString::new("hello").expect("CString::new failed");
        assert_eq!(libc::prctl(libc::PR_SET_NAME, thread_name.as_ptr() as libc::c_long, 0 as libc::c_long, 0 as libc::c_long, 0 as libc::c_long), 0);
        let mut buf = [0; 6];
        assert_eq!(libc::prctl(libc::PR_GET_NAME, buf.as_mut_ptr() as libc::c_long, 0 as libc::c_long, 0 as libc::c_long, 0 as libc::c_long), 0);
        assert_eq!(thread_name.as_bytes_with_nul(), buf);
    }
}
