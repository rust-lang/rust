//@ignore-target: windows # No pthreads on Windows

fn main() {
    let rw = std::cell::UnsafeCell::new(libc::PTHREAD_RWLOCK_INITIALIZER);
    unsafe {
        assert_eq!(libc::pthread_rwlock_rdlock(rw.get()), 0);
        libc::pthread_rwlock_destroy(rw.get()); //~ ERROR: destroyed a locked rwlock
    }
}
