//@ run-crash
//@ check-run-results
//@ only-x86_64-unknown-linux-gnu

use std::ffi::c_void;
use std::{ptr, panic};

unsafe extern "C-unwind" {
    fn pthread_exit(ptr: *mut c_void) -> !;
}

fn main() {
    panic::catch_unwind(|| unsafe {
        pthread_exit(ptr::null_mut());
    }).unwrap();
}
