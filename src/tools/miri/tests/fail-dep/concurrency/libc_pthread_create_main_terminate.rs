//@ignore-target: windows # No pthreads on Windows
//@error-in-other-file: the main thread terminated without waiting for all remaining threads

// Check that we terminate the program when the main thread terminates.

use std::{mem, ptr};

extern "C" fn thread_start(_null: *mut libc::c_void) -> *mut libc::c_void {
    loop {}
}

fn main() {
    unsafe {
        let mut native: libc::pthread_t = mem::zeroed();
        assert_eq!(
            libc::pthread_create(&mut native, ptr::null(), thread_start, ptr::null_mut()),
            0
        );
    }
}
