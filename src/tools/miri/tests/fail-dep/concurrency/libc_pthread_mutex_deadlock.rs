//@ignore-target: windows # No pthreads on Windows
//@error-in-other-file: deadlock
// We are making scheduler assumptions here.
//@compile-flags: -Zmiri-deterministic-concurrency

use std::cell::UnsafeCell;
use std::sync::Arc;
use std::thread;

struct Mutex(UnsafeCell<libc::pthread_mutex_t>);

unsafe impl Send for Mutex {}
unsafe impl Sync for Mutex {}

fn new_lock() -> Arc<Mutex> {
    Arc::new(Mutex(UnsafeCell::new(libc::PTHREAD_MUTEX_INITIALIZER)))
}

fn main() {
    unsafe {
        let lock = new_lock();
        assert_eq!(libc::pthread_mutex_lock(lock.0.get() as *mut _), 0);

        let lock_copy = lock.clone();
        thread::spawn(move || {
            assert_eq!(libc::pthread_mutex_lock(lock_copy.0.get() as *mut _), 0); //~ ERROR: deadlock
        })
        .join()
        .unwrap();
    }
}
