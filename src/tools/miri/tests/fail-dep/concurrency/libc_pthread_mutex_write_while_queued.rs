//@ignore-target: windows # No pthreads on Windows
//@compile-flags: -Zmiri-fixed-schedule

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

// The offset to the "sensitive" part of the mutex (that Miri attaches the metadata to).
const OFFSET: usize = if cfg!(target_os = "macos") { 4 } else { 0 };

fn main() {
    let m = Mutex(UnsafeCell::new(libc::PTHREAD_MUTEX_INITIALIZER));
    thread::scope(|s| {
        // First thread: grabs the lock.
        s.spawn(|| {
            assert_eq!(unsafe { libc::pthread_mutex_lock(m.get()) }, 0);
            thread::yield_now();
            unreachable!();
        });
        // Second thread: queues for the lock.
        s.spawn(|| {
            assert_eq!(unsafe { libc::pthread_mutex_lock(m.get()) }, 0);
            unreachable!();
        });
        // Third thread: tries to overwrite the lock while second thread is queued.
        s.spawn(|| {
            let atomic_ref = unsafe { &*m.get().byte_add(OFFSET).cast::<AtomicU32>() };
            atomic_ref.store(0, Ordering::Relaxed); //~ERROR: write of `pthread_mutex_t` is forbidden while the queue is non-empty
        });
    });
}
