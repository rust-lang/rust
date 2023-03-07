//@ignore-target-windows: No libc on Windows
//@error-pattern: the main thread terminated without waiting for all remaining threads

// Check that we terminate the program when the main thread terminates.

use std::{mem, ptr};

extern "C" fn thread_start(_null: *mut libc::c_void) -> *mut libc::c_void {
    loop {}
}

fn main() {
    unsafe {
        let mut native: libc::pthread_t = mem::zeroed();
        let attr: libc::pthread_attr_t = mem::zeroed();
        // assert_eq!(libc::pthread_attr_init(&mut attr), 0); FIXME: this function is not yet implemented.
        assert_eq!(libc::pthread_create(&mut native, &attr, thread_start, ptr::null_mut()), 0);
    }
}
