//@ignore-target: windows # No pthreads on Windows
//~^ERROR: calling a function with fewer arguments than it requires

//! The thread function must have exactly one argument.

use std::{mem, ptr};

extern "C" fn thread_start(_null: *mut libc::c_void, _x: i32) -> *mut libc::c_void {
    panic!()
}

fn main() {
    unsafe {
        let mut native: libc::pthread_t = mem::zeroed();
        let thread_start: extern "C" fn(*mut libc::c_void, i32) -> *mut libc::c_void = thread_start;
        let thread_start: extern "C" fn(*mut libc::c_void) -> *mut libc::c_void =
            mem::transmute(thread_start);
        assert_eq!(
            libc::pthread_create(&mut native, ptr::null(), thread_start, ptr::null_mut()),
            0
        );
        assert_eq!(libc::pthread_join(native, ptr::null_mut()), 0);
    }
}
