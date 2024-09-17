//@ignore-target: windows # No pthreads on Windows

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
            assert_eq!(libc::pthread_mutex_unlock(lock_copy.0.get() as *mut _), 0); //~ ERROR: Undefined Behavior: unlocked a default mutex that was not locked by the current thread
        })
        .join()
        .unwrap();
    }
}
