//@ignore-target: windows # No pthreads on Windows

fn main() {
    let rw = std::cell::UnsafeCell::new(libc::PTHREAD_RWLOCK_INITIALIZER);
    unsafe {
        libc::pthread_rwlock_unlock(rw.get()); //~ ERROR: was not locked
    }
}
