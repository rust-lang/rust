//@ignore-target-windows: No libc on Windows

//@compile-flags: -Zmiri-disable-abi-check

//! Unwinding past the top frame of a stack is Undefined Behavior.

#![feature(c_unwind)]

use std::{mem, ptr};

extern "C-unwind" fn thread_start(_null: *mut libc::c_void) -> *mut libc::c_void {
    //~^ ERROR: unwinding past the topmost frame of the stack
    panic!()
}

fn main() {
    unsafe {
        let mut native: libc::pthread_t = mem::zeroed();
        let attr: libc::pthread_attr_t = mem::zeroed();
        // assert_eq!(libc::pthread_attr_init(&mut attr), 0); FIXME: this function is not yet implemented.
        // Cast to avoid inserting abort-on-unwind.
        let thread_start: extern "C-unwind" fn(*mut libc::c_void) -> *mut libc::c_void =
            thread_start;
        let thread_start: extern "C" fn(*mut libc::c_void) -> *mut libc::c_void =
            mem::transmute(thread_start);
        assert_eq!(libc::pthread_create(&mut native, &attr, thread_start, ptr::null_mut()), 0);
        assert_eq!(libc::pthread_join(native, ptr::null_mut()), 0);
    }
}
