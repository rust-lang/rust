//@ignore-target: windows # No pthreads on Windows

// Joining an already joined thread is undefined behavior.

use std::{mem, ptr};

extern "C" fn thread_start(_null: *mut libc::c_void) -> *mut libc::c_void {
    ptr::null_mut()
}

fn main() {
    unsafe {
        let mut native: libc::pthread_t = mem::zeroed();
        assert_eq!(
            libc::pthread_create(&mut native, ptr::null(), thread_start, ptr::null_mut()),
            0
        );
        assert_eq!(libc::pthread_join(native, ptr::null_mut()), 0);
        assert_eq!(libc::pthread_join(native, ptr::null_mut()), 0); //~ ERROR: Undefined Behavior: trying to join an already joined thread
    }
}
