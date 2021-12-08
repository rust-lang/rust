// ignore-windows: No libc on Windows
#![feature(rustc_private)]

/// Test that destroying a pthread_rwlock twice fails, even without a check for number validity
extern crate libc;

fn main() {
    unsafe {
        let mut lock = libc::PTHREAD_RWLOCK_INITIALIZER;

        libc::pthread_rwlock_destroy(&mut lock);

        libc::pthread_rwlock_destroy(&mut lock);
        //~^ Undefined Behavior: using uninitialized data, but this operation requires initialized memory
    }
}
