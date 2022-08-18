//@ignore-target-windows: No libc on Windows

#![feature(rustc_private)]

fn main() {
    let rw = std::cell::UnsafeCell::new(libc::PTHREAD_RWLOCK_INITIALIZER);
    unsafe {
        libc::pthread_rwlock_unlock(rw.get()); //~ ERROR: was not locked
    }
}
