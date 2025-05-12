//@only-target: darwin

use std::cell::UnsafeCell;

fn main() {
    let lock = UnsafeCell::new(libc::OS_UNFAIR_LOCK_INIT);

    unsafe { libc::os_unfair_lock_lock(lock.get()) };
    let lock = lock;
    // This needs to either error or deadlock.
    unsafe { libc::os_unfair_lock_lock(lock.get()) };
    //~^ error: deadlock
}
