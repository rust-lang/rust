//@ignore-target: windows # No pthreads on Windows
//@compile-flags: -Zmiri-deterministic-concurrency
//@error-in-other-file: deallocation of `pthread_mutex_t` is forbidden while the queue is non-empty

use std::cell::UnsafeCell;
use std::sync::atomic::*;
use std::thread;

struct Mutex(UnsafeCell<libc::pthread_mutex_t>);
impl Mutex {
    fn get(&self) -> *mut libc::pthread_mutex_t {
        self.0.get()
    }
}

unsafe impl Send for Mutex {}
unsafe impl Sync for Mutex {}

fn main() {
    let m = Box::new(Mutex(UnsafeCell::new(libc::PTHREAD_MUTEX_INITIALIZER)));
    let initialized = AtomicBool::new(false);
    thread::scope(|s| {
        // First thread: initializes the lock, and then grabs it.
        s.spawn(|| {
            // Initialize (so the third thread can happens-after the write that occurs here).
            assert_eq!(unsafe { libc::pthread_mutex_lock(m.get()) }, 0);
            assert_eq!(unsafe { libc::pthread_mutex_unlock(m.get()) }, 0);
            initialized.store(true, Ordering::Release);
            // Grab and hold.
            assert_eq!(unsafe { libc::pthread_mutex_lock(m.get()) }, 0);
            thread::yield_now();
            unreachable!();
        });
        // Second thread: queues for the lock.
        s.spawn(|| {
            assert_eq!(unsafe { libc::pthread_mutex_lock(m.get()) }, 0);
            unreachable!();
        });
        // Third thread: tries to free the lock while second thread is queued.
        s.spawn(|| {
            // Ensure we happen-after the initialization write.
            assert!(initialized.load(Ordering::Acquire));
            // Now drop it.
            drop(unsafe { Box::from_raw(m.get().cast::<Mutex>()) });
        });
    });
    unreachable!();
}
