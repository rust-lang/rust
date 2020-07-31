// ignore-windows: Concurrency on Windows is not supported yet.
// error-pattern: unwinding past the topmost frame of the stack

//! Unwinding past the top frame of a stack is Undefined Behavior.

#![feature(rustc_private)]

extern crate libc;

use std::{mem, ptr};

extern "C" fn thread_start(_null: *mut libc::c_void) -> *mut libc::c_void {
    panic!()
}

fn main() {
    unsafe {
        let mut native: libc::pthread_t = mem::zeroed();
        let attr: libc::pthread_attr_t = mem::zeroed();
        // assert_eq!(libc::pthread_attr_init(&mut attr), 0); FIXME: this function is not yet implemented.
        assert_eq!(libc::pthread_create(&mut native, &attr, thread_start, ptr::null_mut()), 0);
        assert_eq!(libc::pthread_join(native, ptr::null_mut()), 0);
    }
}
