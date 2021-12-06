// ignore-windows: No libc on Windows
#![feature(rustc_private)]

/// Test that destroying a pthread_mutex twice fails, even without a check for number validity
extern crate libc;

fn main() {
    unsafe {
        use core::mem::MaybeUninit;

        let mut attr = MaybeUninit::<libc::pthread_mutexattr_t>::uninit();
        libc::pthread_mutexattr_init(attr.as_mut_ptr());

        let mut mutex = MaybeUninit::<libc::pthread_mutex_t>::uninit();

        libc::pthread_mutex_init(mutex.as_mut_ptr(), attr.as_ptr());

        libc::pthread_mutex_destroy(mutex.as_mut_ptr());

        libc::pthread_mutex_destroy(mutex.as_mut_ptr());
        //~^ Undefined Behavior: using uninitialized data, but this operation requires initialized memory
    }
}
