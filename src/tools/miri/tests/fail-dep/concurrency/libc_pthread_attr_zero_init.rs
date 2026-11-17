//@ignore-target: windows # No pthreads on Windows

/// Test that using a pthread_attr that was never initialized fails.
use std::{mem, ptr};

extern "C" fn thread_start(_arg: *mut libc::c_void) -> *mut libc::c_void {
    ptr::null_mut()
}

fn main() {
    unsafe {
        let attr: libc::pthread_attr_t = mem::zeroed();
        let mut thread: libc::pthread_t = mem::zeroed();
        libc::pthread_create(&mut thread, &attr, thread_start, ptr::null_mut());
        //~^ ERROR: has not been initialized
    }
}
