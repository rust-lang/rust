//@only-target: darwin
#![feature(sync_unsafe_cell)]

use std::cell::SyncUnsafeCell;
use std::sync::atomic::*;
use std::thread;

fn main() {
    let lock = SyncUnsafeCell::new(libc::OS_UNFAIR_LOCK_INIT);

    thread::scope(|s| {
        // First thread: grabs the lock.
        s.spawn(|| {
            unsafe { libc::os_unfair_lock_lock(lock.get()) };
            thread::yield_now();
            unreachable!();
        });
        // Second thread: queues for the lock.
        s.spawn(|| {
            unsafe { libc::os_unfair_lock_lock(lock.get()) };
            unreachable!();
        });
        // Third thread: tries to read the lock while second thread is queued.
        s.spawn(|| {
            let atomic_ref = unsafe { &*lock.get().cast::<AtomicU32>() };
            let _val = atomic_ref.load(Ordering::Relaxed); //~ERROR: read of `os_unfair_lock` is forbidden while the queue is non-empty
        });
    });
}
