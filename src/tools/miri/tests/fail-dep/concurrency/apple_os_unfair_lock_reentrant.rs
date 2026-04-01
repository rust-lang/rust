//@only-target: darwin

use std::cell::UnsafeCell;

fn main() {
    let lock = UnsafeCell::new(libc::OS_UNFAIR_LOCK_INIT);

    unsafe {
        libc::os_unfair_lock_lock(lock.get());
        libc::os_unfair_lock_lock(lock.get());
        //~^ error: abnormal termination: attempted to lock an os_unfair_lock that is already locked by the current thread
    }
}
