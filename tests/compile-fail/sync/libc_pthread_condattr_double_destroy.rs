// ignore-windows: No libc on Windows
#![feature(rustc_private)]

/// Test that destroying a pthread_condattr twice fails, even without a check for number validity
extern crate libc;

fn main() {
    unsafe {
        use core::mem::MaybeUninit;
        let mut attr = MaybeUninit::<libc::pthread_condattr_t>::uninit();

        libc::pthread_condattr_init(attr.as_mut_ptr());

        libc::pthread_condattr_destroy(attr.as_mut_ptr());

        libc::pthread_condattr_destroy(attr.as_mut_ptr());
        //~^ Undefined Behavior: using uninitialized data, but this operation requires initialized memory
    }
}
