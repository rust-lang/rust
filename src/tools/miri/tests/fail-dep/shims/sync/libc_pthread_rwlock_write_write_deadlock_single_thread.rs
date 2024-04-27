//@ignore-target-windows: No libc on Windows

fn main() {
    let rw = std::cell::UnsafeCell::new(libc::PTHREAD_RWLOCK_INITIALIZER);
    unsafe {
        assert_eq!(libc::pthread_rwlock_wrlock(rw.get()), 0);
        libc::pthread_rwlock_wrlock(rw.get()); //~ ERROR: deadlock
    }
}
